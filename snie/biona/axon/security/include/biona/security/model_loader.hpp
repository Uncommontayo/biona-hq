#pragma once

/**
 * @file model_loader.hpp
 * @brief Secure ONNX model loader — in-memory only, plaintext never touches disk.
 *
 * SECURITY CRITICAL:
 *   1. VERIFY HMAC BEFORE DECRYPTION.
 *      Reason: prevents padding oracle and chosen-ciphertext attacks.
 *              An attacker who can trigger decryption of chosen ciphertexts
 *              can recover the key if we decrypt before checking integrity.
 *
 *   2. VERIFY GCM AUTH TAG BEFORE USING ANY PLAINTEXT BYTES.
 *      Reason: AES-GCM provides authenticated encryption; using unauthenticated
 *              plaintext violates the security model and may expose the engine
 *              to byte-by-byte oracle attacks.
 *
 *   3. mlock() the output buffer IMMEDIATELY after allocation.
 *      Reason: prevents the OS from paging decrypted model bytes to swap,
 *              which would leave plaintext ONNX weights on disk.
 *
 *   4. NEVER LOG key material, IV, auth tag, or any portion of plaintext.
 */

#include "biona/security/secret_manager.hpp"
#include "biona/security/secure_buffer.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace biona {

// ---------------------------------------------------------------------------
// EncryptedModelBundle — binary format for .biona bundle files
// ---------------------------------------------------------------------------

/**
 * @brief Parsed representation of an encrypted Biona model bundle.
 *
 * On-disk binary layout (little-endian):
 *   [4 bytes]  magic        — "BION" = {0x42, 0x49, 0x4F, 0x4E}
 *   [4 bytes]  version      — uint32_t, currently 1
 *   [12 bytes] iv           — AES-GCM 96-bit initialisation vector
 *   [16 bytes] tag          — AES-GCM 128-bit authentication tag
 *   [32 bytes] hmac         — HMAC-SHA256 over (iv || tag || ciphertext)
 *   [N bytes]  ciphertext   — AES-256-GCM encrypted ONNX model bytes
 */
struct EncryptedModelBundle {
    std::array<uint8_t, 4>  magic;     ///< Must equal {0x42, 0x49, 0x4F, 0x4E}
    uint32_t                version;   ///< Bundle format version (must be 1)
    std::array<uint8_t, 12> iv;        ///< AES-GCM 96-bit IV (random, unique per bundle)
    std::array<uint8_t, 16> tag;       ///< AES-GCM 128-bit authentication tag
    std::array<uint8_t, 32> hmac;      ///< HMAC-SHA256(iv || tag || ciphertext)
    std::vector<uint8_t>    ciphertext;///< AES-256-GCM encrypted ONNX bytes

    /**
     * @brief Read and parse a .biona bundle from disk.
     *
     * @param path  Path to the encrypted bundle file.
     * @return Parsed bundle struct.
     * @throws BionaException(ModelDecryptFailed) if magic or version is invalid.
     * @throws std::filesystem::filesystem_error on I/O failure.
     */
    static EncryptedModelBundle fromFile(const std::filesystem::path& path);
};

// ---------------------------------------------------------------------------
// ModelLoader
// ---------------------------------------------------------------------------

/**
 * @brief Loads and decrypts an encrypted Biona model bundle into locked RAM.
 *
 * Usage:
 * @code
 *   ModelLoader loader;
 *   SecureBuffer plaintext = loader.load("model.biona", *secret_manager);
 *   // Pass plaintext.data() / plaintext.size() to Ort::Session::CreateFromArray
 * @endcode
 *
 * Thread safety: ModelLoader holds no state — multiple threads may call load()
 * concurrently on the same instance.
 */
class ModelLoader {
public:
    ModelLoader() = default;

    /**
     * @brief Load, verify, and decrypt an encrypted model bundle.
     *
     * Steps (in order — do NOT reorder):
     *   1. Read and parse the EncryptedModelBundle from disk.
     *   2. Derive the AES-256 key from BIONA_SECRET_MODEL_ENCRYPTION_KEY via HKDF-SHA256.
     *   3. Derive the HMAC key from BIONA_SECRET_MODEL_HMAC_KEY.
     *   4. Verify HMAC-SHA256 over (iv || tag || ciphertext) — BEFORE decryption.
     *   5. AES-256-GCM decrypt — verify auth tag before returning any bytes.
     *   6. Return plaintext in a mlock()-ed SecureBuffer.
     *
     * @param bundle_path  Path to the .biona encrypted bundle.
     * @param secrets      SecretManager to retrieve key material.
     * @return SecureBuffer containing the plaintext ONNX model bytes.
     *         The buffer is mlock()-ed and will be zeroed on destruction.
     * @throws BionaException(ModelDecryptFailed) on HMAC or GCM verification failure.
     * @throws BionaException(InitFailed) on I/O or key-derivation failure.
     * @throws BionaException(OOM) if mlock() fails.
     */
    SecureBuffer load(const std::filesystem::path& bundle_path,
                      const SecretManager&         secrets);

private:
    /**
     * @brief AES-256-GCM decrypt and authenticate.
     *
     * SECURITY: Verifies the GCM auth tag before returning any plaintext bytes.
     *           If EVP_DecryptFinal_ex fails, the output buffer is zeroed
     *           and a BionaException(ModelDecryptFailed) is thrown.
     *
     * @param bundle    Parsed bundle containing iv, tag, and ciphertext.
     * @param key_hex   64-char hex string of the 256-bit AES key.
     * @return SecureBuffer with plaintext ONNX model.
     */
    SecureBuffer decryptAESGCM(const EncryptedModelBundle& bundle,
                               const std::string&          key_hex);
};

} // namespace biona
