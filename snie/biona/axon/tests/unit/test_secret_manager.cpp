#include "biona/security/secret_manager.hpp"

#include <cassert>
#include <cstdlib>

// SECURITY NOTE: This test ONLY calls has() — it never prints or asserts
// on actual secret values. Secret material must never appear in test output.

int main() {
    // Set a benign test env var
    const char* test_key = "BIONA_TEST_SECRET_PRESENT";
#if defined(_WIN32)
    _putenv_s("BIONA_TEST_SECRET_PRESENT", "1");
#else
    setenv("BIONA_TEST_SECRET_PRESENT", "1", 1);
#endif

    auto mgr = biona::createSecretManager("env");

    // has() must return true for existing env var
    assert(mgr->has("BIONA_TEST_SECRET_PRESENT") == true);

    // has() must return false for a var that definitely does not exist
    assert(mgr->has("BIONA_KEY_THAT_DOES_NOT_EXIST_XYZ123") == false);

    // Unknown backend type must throw
    bool threw = false;
    try {
        biona::createSecretManager("unknown_backend");
    } catch (const biona::BionaException& e) {
        assert(e.code() == biona::BionaError::UnsupportedRuntime);
        threw = true;
    }
    assert(threw && "createSecretManager with unknown type must throw");

    return 0;
}
