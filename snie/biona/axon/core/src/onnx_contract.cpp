#include "biona/core/onnx_contract.hpp"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace biona {

bool ONNXContractValidator::validate(
    const std::vector<std::string>& actual_inputs,
    const std::vector<std::string>& actual_outputs)
{
    bool ok = true;

    // Expected names derived from contract constants
    const std::vector<std::string_view> expected_inputs = {
        ONNX_INPUT_CHUNK,
        ONNX_INPUT_MEMORY,
        ONNX_INPUT_LC_KEY,
        ONNX_INPUT_LC_VAL,
    };
    const std::vector<std::string_view> expected_outputs = {
        ONNX_OUTPUT_LOGITS,
        ONNX_OUTPUT_MEMORY,
        ONNX_OUTPUT_LC_KEY,
        ONNX_OUTPUT_LC_VAL,
        ONNX_OUTPUT_EMBEDDING,
    };

    auto check = [&](const std::vector<std::string_view>& expected,
                     const std::vector<std::string>&      actual,
                     const char* kind) {
        for (const auto& name : expected) {
            bool found = std::find(actual.begin(), actual.end(),
                                   std::string(name)) != actual.end();
            if (!found) {
                std::cerr << "[BIONA CONTRACT MISMATCH] "
                          << kind << " tensor \"" << name
                          << "\" is required by onnx_contract.hpp "
                          << "(contract version " << ONNX_CONTRACT_VERSION << ") "
                          << "but was NOT found in the loaded model.\n"
                          << "  -> Update biona/lab to export this tensor name, "
                          << "or check for a model/contract version mismatch.\n";
                ok = false;
            }
        }
    };

    check(expected_inputs,  actual_inputs,  "Input");
    check(expected_outputs, actual_outputs, "Output");

    return ok;
}

} // namespace biona
