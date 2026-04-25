#include "biona/core/interfaces/inference_engine.hpp"
#include "biona/core/errors.hpp"

#include <unordered_map>

namespace biona {

std::unordered_map<EngineType, EngineCreator>& InferenceEngineFactory::registry() {
    static std::unordered_map<EngineType, EngineCreator> reg;
    return reg;
}

void InferenceEngineFactory::registerEngine(EngineType type, EngineCreator creator) {
    registry()[type] = std::move(creator);
}

std::unique_ptr<InferenceEngine> InferenceEngineFactory::create(EngineType type,
                                                                const ModelConfig& cfg) {
    auto& reg = registry();
    auto it = reg.find(type);
    throwIf(it == reg.end(),
            BionaError::UnsupportedRuntime,
            "Requested EngineType is not registered on this platform");

    auto engine = it->second();
    throwIf(!engine,
            BionaError::InitFailed,
            "EngineCreator returned nullptr");

    bool ok = engine->initialize(cfg);
    throwIf(!ok, BionaError::InitFailed, "InferenceEngine::initialize() returned false");

    return engine;
}

} // namespace biona
