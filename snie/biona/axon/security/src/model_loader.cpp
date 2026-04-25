#include "biona/security/model_loader.hpp"
#include "biona/security/safe_log.hpp"
#include "biona/core/errors.hpp"

#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/kdf.h>
#include <openssl/sha.h>
#include <openssl/params.h>

// Suppress OpenSSL 3.0 deprecation warnings for HMAC_CTX API on MSVC.
// The EVP_MAC API is used where available; the legacy API remains for clarity.
#ifdef _MSC_VER
#  pragma warning(disable: 4996)
#endif

#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace biona {

// ---------------------------------------------------------------------------
// EncryptedModelBundle::fromFile
// ---------------------------------------------------------------------------

static constexpr std::array<uint8_t, 4> BIONA_MAGIC = {0x42, 0x49, 0x4F, 0x4E};
static constexpr uint32_t               BUNDLE_VERSION = 1;

EncryptedModelBundle EncryptedModelBundle::fromFile(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    throwIf(!f.is_open(), BionaError::ModelDecryptFailed,
            "Cannot open model bundle file");

    EncryptedModelBundle b;

    f.read(reinterpret_cast<char*>(b.magic.data()), 4);
    throwIf(b.magic != BIONA_MAGIC,
            BionaError::ModelDecryptFailed,
            "Invalid model bundle magic bytes (expected 'BION')");

    f.read(reinterpret_cast<char*>(&b.version), sizeof(b.version));
    throwIf(b.version != BUNDLE_VERSION,
            BionaError::ModelDecryptFailed,
            "Unsupported model bundle version");

    f.read(reinterpret_cast<char*>(b.iv.data()),   b.iv.size());
    f.read(reinterpret_cast<char*>(b.tag.data()),  b.tag.size());
    f.read(reinterpret_cast<char*>(b.hmac.data()), b.hmac.size());

    throwIf(!f.good(), BionaError::ModelDecryptFailed,
            "Truncated bundle header");

    // Read remaining bytes as ciphertext
    b.ciphertext = std::vector<uint8_t>(
        (std::istreambuf_iterator<char>(f)),
        std::istreambuf_iterator<char>());

    throwIf(b.ciphertext.empty(), BionaError::ModelDecryptFailed,
            "Bundle contains empty ciphertext");
    return b;
}

// ---------------------------------------------------------------------------
// Hex decode helper
// ---------------------------------------------------------------------------

static std::vector<uint8_t> hexDecode(const std::string& hex) {
    throwIf(hex.size() % 2 != 0, BionaError::ModelDecryptFailed,
            "Key hex string has odd length");
    std::vector<uint8_t> out(hex.size() / 2);
    for (size_t i = 0; i < out.size(); ++i) {
        auto nibble = [](char c) -> uint8_t {
            if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
            if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(c - 'a' + 10);
            if (c >= 'A' && c <= 'F') return static_cast<uint8_t>(c - 'A' + 10);
            throw BionaException(BionaError::ModelDecryptFailed, "Invalid hex character in key");
            return 0;
        };
        out[i] = static_cast<uint8_t>((nibble(hex[2*i]) << 4) | nibble(hex[2*i+1]));
    }
    return out;
}

// ---------------------------------------------------------------------------
// HMAC-SHA256 verification
// ---------------------------------------------------------------------------

static void verifyHMAC(const EncryptedModelBundle& bundle, const std::string& hmac_key_hex) {
    // Derive HMAC key from hex
    auto hmac_key = hexDecode(hmac_key_hex);

    // Compute HMAC-SHA256 over (iv || tag || ciphertext)
    std::array<uint8_t, 32> computed{};
    unsigned int len = 32;

    HMAC_CTX* ctx = HMAC_CTX_new();
    throwIf(!ctx, BionaError::InferenceFailed, "HMAC_CTX_new failed");

    bool ok = HMAC_Init_ex(ctx, hmac_key.data(), static_cast<int>(hmac_key.size()),
                           EVP_sha256(), nullptr);
    ok = ok && HMAC_Update(ctx, bundle.iv.data(),         bundle.iv.size());
    ok = ok && HMAC_Update(ctx, bundle.tag.data(),        bundle.tag.size());
    ok = ok && HMAC_Update(ctx, bundle.ciphertext.data(), bundle.ciphertext.size());
    ok = ok && HMAC_Final(ctx, computed.data(), &len);
    HMAC_CTX_free(ctx);

    throwIf(!ok, BionaError::ModelDecryptFailed, "HMAC computation failed");

    // Constant-time comparison
    uint8_t diff = 0;
    for (size_t i = 0; i < 32; ++i) {
        diff |= computed[i] ^ bundle.hmac[i];
    }

    // SECURITY: Wipe computed HMAC before throwing
    volatile uint8_t* p = computed.data();
    for (size_t i = 0; i < 32; ++i) p[i] = 0;

    throwIf(diff != 0, BionaError::ModelDecryptFailed,
            "Model bundle HMAC verification failed — bundle may be tampered");
}

// ---------------------------------------------------------------------------
// ModelLoader::decryptAESGCM
// ---------------------------------------------------------------------------

SecureBuffer ModelLoader::decryptAESGCM(const EncryptedModelBundle& bundle,
                                        const std::string& key_hex) {
    auto key_bytes = hexDecode(key_hex);
    throwIf(key_bytes.size() != 32, BionaError::ModelDecryptFailed,
            "AES-256 key must be exactly 32 bytes (64 hex chars)");

    SecureBuffer plaintext(bundle.ciphertext.size()); // Allocates and mlocks

    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    throwIf(!ctx, BionaError::InferenceFailed, "EVP_CIPHER_CTX_new failed");

    bool ok = EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr);
    ok = ok && EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN,
                                   static_cast<int>(bundle.iv.size()), nullptr);
    ok = ok && EVP_DecryptInit_ex(ctx, nullptr, nullptr,
                                  key_bytes.data(), bundle.iv.data());
    throwIf(!ok, BionaError::ModelDecryptFailed, "AES-GCM init failed");

    int out_len = 0;
    ok = EVP_DecryptUpdate(ctx,
                           plaintext.data(), &out_len,
                           bundle.ciphertext.data(),
                           static_cast<int>(bundle.ciphertext.size()));
    throwIf(!ok, BionaError::ModelDecryptFailed, "AES-GCM decrypt update failed");

    // Set the expected GCM tag BEFORE calling Final
    ok = EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, 16,
                              const_cast<uint8_t*>(bundle.tag.data()));
    throwIf(!ok, BionaError::ModelDecryptFailed, "AES-GCM set tag failed");

    // SECURITY: EVP_DecryptFinal_ex verifies the auth tag.
    // If it fails, DO NOT use any bytes from plaintext.
    int final_len = 0;
    int final_ok = EVP_DecryptFinal_ex(ctx, plaintext.data() + out_len, &final_len);
    EVP_CIPHER_CTX_free(ctx);

    if (final_ok != 1) {
        // Zero the partially decrypted buffer before throwing
        volatile uint8_t* wipe = plaintext.data();
        for (size_t i = 0; i < plaintext.size(); ++i) wipe[i] = 0;
        throw BionaException(BionaError::ModelDecryptFailed,
            "AES-GCM authentication tag verification failed — "
            "model plaintext was NOT returned");
    }

    return plaintext;
}

// ---------------------------------------------------------------------------
// ModelLoader::load
// ---------------------------------------------------------------------------

SecureBuffer ModelLoader::load(const std::filesystem::path& bundle_path,
                               const SecretManager& secrets) {
    SafeLog::event(BIONA_EVENT_MODEL_LOAD_SUCCESS); // Will be overwritten on failure

    // 1. Read and parse bundle
    EncryptedModelBundle bundle = EncryptedModelBundle::fromFile(bundle_path);

    // 2. Retrieve key material — wrapped in SecureString to ensure zeroing
    SecureString enc_key(secrets.get(BIONA_SECRET_MODEL_ENCRYPTION_KEY));
    SecureString hmac_key(secrets.get(BIONA_SECRET_MODEL_HMAC_KEY));

    // 3. Verify HMAC BEFORE decryption (prevents padding oracle / CCA)
    try {
        verifyHMAC(bundle, hmac_key.str());
    } catch (...) {
        SafeLog::event(BIONA_EVENT_MODEL_LOAD_FAILED);
        throw;
    }

    // 4+5. AES-256-GCM decrypt (GCM tag verified inside decryptAESGCM)
    SecureBuffer plaintext = decryptAESGCM(bundle, enc_key.str());

    SafeLog::event(BIONA_EVENT_MODEL_LOAD_SUCCESS);
    return plaintext;
}

} // namespace biona
