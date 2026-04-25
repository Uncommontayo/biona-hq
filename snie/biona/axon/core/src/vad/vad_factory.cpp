#include "biona/core/interfaces/vad.hpp"
#include "biona/core/errors.hpp"
#include "webrtc_vad.hpp"
#include "energy_vad.hpp"

#include <memory>

namespace biona {

std::unique_ptr<VAD> createVAD(const VADConfig& cfg) {
    switch (cfg.type) {
        case VADType::WEBRTC:
            return std::make_unique<WebRTCVAD>(cfg);
        case VADType::ENERGY:
            return std::make_unique<EnergyVAD>(cfg);
        case VADType::SILERO:
            throw BionaException(BionaError::UnsupportedRuntime,
                "Silero VAD is not yet implemented");
    }
    throw BionaException(BionaError::UnsupportedRuntime, "Unknown VADType");
}

} // namespace biona
