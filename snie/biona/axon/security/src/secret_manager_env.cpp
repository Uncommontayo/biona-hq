#include "biona/security/secret_manager.hpp"

#ifdef _MSC_VER
#  pragma warning(disable: 4996) // 'getenv': use _dupenv_s — intentional, cross-platform code
#endif

#include <cstdlib>
#include <string>
#include <string_view>

namespace biona {

/**
 * @brief SecretManager implementation that reads from process environment variables.
 *
 * Key lookup: if a non-empty prefix is configured, it is prepended to the key
 * before the getenv() call.
 *
 * Example with prefix "BIONA_":
 *   get("MODEL_ENCRYPTION_KEY") → reads env var "BIONA_MODEL_ENCRYPTION_KEY"
 */
class EnvSecretManager : public SecretManager {
public:
    explicit EnvSecretManager(std::string prefix = "")
        : prefix_(std::move(prefix))
    {}

    std::string get(std::string_view key) const override {
        std::string env_key;
        env_key.reserve(prefix_.size() + key.size());
        env_key += prefix_;
        env_key += key;

        const char* val = std::getenv(env_key.c_str());
        if (val == nullptr) {
            throw BionaException(
                BionaError::InitFailed,
                std::string("Required environment variable not set: ") + env_key);
        }
        return std::string(val);
    }

    bool has(std::string_view key) const noexcept override {
        try {
            std::string env_key;
            env_key.reserve(prefix_.size() + key.size());
            env_key += prefix_;
            env_key += key;
            return std::getenv(env_key.c_str()) != nullptr;
        } catch (...) {
            return false;
        }
    }

private:
    std::string prefix_;
};

} // namespace biona

// Make EnvSecretManager available to the factory
namespace biona {
namespace detail {

std::unique_ptr<SecretManager> makeEnvSecretManager() {
    return std::make_unique<EnvSecretManager>(); // no prefix by default
}

} // namespace detail
} // namespace biona
