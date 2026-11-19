import CoreML
import Foundation

/// Isolated probe policy: detach values from a producer's buffers before another model is loaded/run.
nonisolated enum ProbeOwnedTensorCopy {
    static let policy = "owned-fp32-between-models-v1"
    private static func invalid(_ message: String) -> NSError {
        NSError(domain: "OwnedTensorProbe", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
    static func float32(_ source: MLMultiArray) throws -> MLMultiArray {
        let normalized = try SongFormerTensorBridge.float32(source)
        let result = try MLMultiArray(shape: normalized.shape, dataType: .float32)
        guard result.dataPointer != normalized.dataPointer else { throw invalid("Tensor copy aliases producer storage") }
        let values = normalized.dataPointer.assumingMemoryBound(to: Float.self)
        let shape = normalized.shape.map(\.intValue)
        withExtendedLifetime(normalized) {
        result.withUnsafeMutableBufferPointer(ofType: Float.self) { destination, strides in
            var expected = 1, dense = true
            for axis in shape.indices.reversed() {
                if shape[axis] > 1 && strides[axis] != expected { dense = false }
                expected *= shape[axis]
            }
            if dense {
                destination.baseAddress!.update(from: values, count: normalized.count)
            } else {
                for linear in 0..<normalized.count {
                    var remaining = linear, offset = 0
                    for axis in shape.indices.reversed() {
                        offset += (remaining % shape[axis])*strides[axis]
                        remaining /= shape[axis]
                    }
                    destination[offset] = values[linear]
                }
            }
        }
        }
        return result
    }
    /// Native tests: a retained pass-through object fails the mutation test below.
    static func selfTest() throws {
        let source = try MLMultiArray(shape: [2,3], dataType: .float32)
        let original: [Float] = [0.12345679, -1.234567, 12345.678, -0.0, Float.leastNonzeroMagnitude, Float.greatestFiniteMagnitude]
        for i in original.indices { source[i] = NSNumber(value: original[i]) }
        let clone = try float32(source)
        guard clone.shape == source.shape, clone.dataType == .float32, clone !== source else { throw invalid("Copy contract differs") }
        for i in 0..<6 {
            guard clone[i].floatValue.bitPattern == source[i].floatValue.bitPattern else { throw invalid("FP32 copy changes bits") }
        }
        source[0] = 999
        guard clone[0].floatValue == original[0] else { throw invalid("Producer mutation changed the copy") }
        clone[1] = -999
        guard source[1].floatValue == original[1] else { throw invalid("Consumer mutation changed the producer") }
        let half = try MLMultiArray(shape: [2], dataType: .float16)
        half[0] = 1.5; half[1] = -2.25
        let converted = try float32(half)
        half[0] = 3
        guard converted[0].floatValue == 1.5, converted[1].floatValue == -2.25 else { throw invalid("Half conversion/ownership differs") }
        let backing = UnsafeMutablePointer<Float>.allocate(capacity: 6)
        backing.initialize(repeating: 0, count: 6)
        defer { backing.deinitialize(count: 6); backing.deallocate() }
        for i in 0..<6 { backing[i] = Float(i)+0.25 }
        let view = try MLMultiArray(dataPointer: UnsafeMutableRawPointer(backing), shape: [2,2], dataType: .float32,
            strides: [3,1], deallocator: { _ in })
        let detached = try float32(view)
        let expected: [Float] = [0.25,1.25,3.25,4.25]
        for i in expected.indices {
            guard detached[i].floatValue == expected[i] else { throw invalid("Strided tensor logical order changed") }
        }
        backing[3] = -123
        guard detached[2].floatValue == 3.25 else { throw invalid("Strided copy aliases producer storage") }
        let invalidSource = try MLMultiArray(shape: [1], dataType: .float32)
        invalidSource[0] = NSNumber(value: Float.nan)
        var rejected = false
        do { _ = try float32(invalidSource) } catch { rejected = true }
        guard rejected else { throw invalid("Non-finite tensor accepted") }
    }
}
