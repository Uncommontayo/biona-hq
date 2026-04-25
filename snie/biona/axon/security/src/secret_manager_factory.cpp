#include "biona/security/secret_manager.hpp"
#include "biona/core/errors.hpp"

#include <memory>
#include <string>

namespace biona {

namespace detail {
// Defined in secret_manager_env.cpp
std::unique_ptr<SecretManager> makeEnvSecretManager();
} // namespace detail

std::unique_ptr<SecretManager> createSecretManager(const std::string& type) {
    if (type == "env") {
        return detail::makeEnvSecretManager();
    }
    if (type == "keychain") {
        throw BionaException(BionaError::UnsupportedRuntime,
            "SecretManager type 'keychain' not yet implemented "
            "(iOS Keychain implementation pending)");
    }
    if (type == "keystore") {
        throw BionaException(BionaError::UnsupportedRuntime,
            "SecretManager type 'keystore' not yet implemented "
            "(Android Keystore implementation pending)");
    }
    if (type == "tpm") {
        throw BionaException(BionaError::UnsupportedRuntime,
            "SecretManager type 'tpm' not yet implemented "
            "(Linux TPM implementation pending)");
    }
    throw BionaException(BionaError::UnsupportedRuntime,
        std::string("Unknown SecretManager type: ") + type);
}

} // namespace biona
