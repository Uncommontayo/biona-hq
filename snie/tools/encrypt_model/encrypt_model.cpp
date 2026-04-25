/**
 * @file encrypt_model.cpp
 * @brief CLI tool — encrypt a plaintext .onnx file into a .biona bundle.
 *
 * Usage:
 *   encrypt_model <input.onnx> <output.biona>
 *
 * Required environment variables:
 *   MODEL_ENCRYPTION_KEY  — 64-char hex string (256-bit AES key)
 *   MODEL_HMAC_KEY        — 64-char hex string (256-bit HMAC key)
 *
 * This tool is used at MODEL PACKAGING TIME ONLY, never at runtime on-device.
 * Run it in a secure build environment; do not commit output bundles to source control.
 */

#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/rand.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<uint8_t> hexDecode(const char* hex_env, const char* var_name) {
    const char* val = std::getenv(hex_env);
    if (!val) {
        std::cerr << "ERROR: environment variable " << var_name << " is not set.\n";
        std::exit(1);
    }
    std::string h(val);
    if (h.size() != 64) {
        std::cerr << "ERROR: " << var_name << " must be 64 hex characters (256-bit key).\n";
        std::exit(1);
    }
    std::vector<uint8_t> out(32);
    for (size_t i = 0; i < 32; ++i) {
        auto nibble = [](char c) -> uint8_t {
            if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
            if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(c - 'a' + 10);
            if (c >= 'A' && c <= 'F') return static_cast<uint8_t>(c - 'A' + 10);
            throw std::runtime_error("Invalid hex char in key");
        };
        out[i] = static_cast<uint8_t>((nibble(h[2*i]) << 4) | nibble(h[2*i+1]));
    }
    return out;
}

static void writeLE32(std::ostream& out, uint32_t v) {
    uint8_t buf[4] = {
        static_cast<uint8_t>(v & 0xFF),
        static_cast<uint8_t>((v >> 8) & 0xFF),
        static_cast<uint8_t>((v >> 16) & 0xFF),
        static_cast<uint8_t>((v >> 24) & 0xFF),
    };
    out.write(reinterpret_cast<const char*>(buf), 4);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: encrypt_model <input.onnx> <output.biona>\n";
        return 1;
    }

    // 1. Load plaintext ONNX
    std::ifstream onnx_file(argv[1], std::ios::binary);
    if (!onnx_file) {
        std::cerr << "ERROR: Cannot open input file: " << argv[1] << '\n';
        return 1;
    }
    std::vector<uint8_t> plaintext(
        (std::istreambuf_iterator<char>(onnx_file)),
        std::istreambuf_iterator<char>());
    if (plaintext.empty()) {
        std::cerr << "ERROR: Input ONNX file is empty.\n";
        return 1;
    }

    // 2. Read keys from environment
    auto enc_key  = hexDecode("MODEL_ENCRYPTION_KEY", "MODEL_ENCRYPTION_KEY");
    auto hmac_key = hexDecode("MODEL_HMAC_KEY",       "MODEL_HMAC_KEY");

    // 3. Generate random 12-byte IV
    std::array<uint8_t, 12> iv{};
    if (RAND_bytes(iv.data(), 12) != 1) {
        std::cerr << "ERROR: RAND_bytes failed.\n";
        return 1;
    }

    // 4. AES-256-GCM encrypt
    std::vector<uint8_t> ciphertext(plaintext.size());
    std::array<uint8_t, 16> tag{};
    {
        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr);
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, 12, nullptr);
        EVP_EncryptInit_ex(ctx, nullptr, nullptr, enc_key.data(), iv.data());

        int out_len = 0;
        EVP_EncryptUpdate(ctx, ciphertext.data(), &out_len,
                          plaintext.data(), static_cast<int>(plaintext.size()));

        int final_len = 0;
        EVP_EncryptFinal_ex(ctx, ciphertext.data() + out_len, &final_len);
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, 16, tag.data());
        EVP_CIPHER_CTX_free(ctx);
    }

    // 5. Compute HMAC-SHA256 over (iv || tag || ciphertext)
    std::array<uint8_t, 32> hmac_val{};
    {
        unsigned int len = 32;
        HMAC_CTX* ctx = HMAC_CTX_new();
        HMAC_Init_ex(ctx, hmac_key.data(), static_cast<int>(hmac_key.size()),
                     EVP_sha256(), nullptr);
        HMAC_Update(ctx, iv.data(),         iv.size());
        HMAC_Update(ctx, tag.data(),        tag.size());
        HMAC_Update(ctx, ciphertext.data(), ciphertext.size());
        HMAC_Final(ctx, hmac_val.data(), &len);
        HMAC_CTX_free(ctx);
    }

    // 6. Write bundle
    std::ofstream out(argv[2], std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "ERROR: Cannot open output file: " << argv[2] << '\n';
        return 1;
    }

    // magic "BION"
    const uint8_t magic[4] = {0x42, 0x49, 0x4F, 0x4E};
    out.write(reinterpret_cast<const char*>(magic), 4);
    writeLE32(out, 1); // version
    out.write(reinterpret_cast<const char*>(iv.data()),         12);
    out.write(reinterpret_cast<const char*>(tag.data()),        16);
    out.write(reinterpret_cast<const char*>(hmac_val.data()),   32);
    out.write(reinterpret_cast<const char*>(ciphertext.data()), static_cast<std::streamsize>(ciphertext.size()));

    if (!out.good()) {
        std::cerr << "ERROR: Write to output file failed.\n";
        return 1;
    }

    std::cout << "Encrypted model written to: " << argv[2] << '\n'
              << "  Input size:  " << plaintext.size()   << " bytes\n"
              << "  Output size: " << ciphertext.size()  << " bytes (+ 68-byte header)\n";

    // Zero sensitive data before exit
    volatile uint8_t* pe = enc_key.data();
    volatile uint8_t* ph = hmac_key.data();
    for (size_t i = 0; i < 32; ++i) { pe[i] = 0; ph[i] = 0; }
    volatile uint8_t* pp = plaintext.data();
    for (size_t i = 0; i < plaintext.size(); ++i) pp[i] = 0;

    return 0;
}
