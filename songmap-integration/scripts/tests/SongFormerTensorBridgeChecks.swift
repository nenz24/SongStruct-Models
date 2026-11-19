import CoreML
import Foundation

@main struct SongFormerTensorBridgeChecks {
    /// -O builds drop precondition messages; print the failing check before exiting.
    static func fail(_ message: String, file: StaticString = #file, line: UInt = #line) -> Never {
        FileHandle.standardError.write(Data("CHECK FAILED (\(file):\(line)): \(message)\n".utf8)); exit(1)
    }
    static func check(_ condition: @autoclosure () -> Bool, _ message: @autoclosure () -> String = "condition", file: StaticString = #file, line: UInt = #line) {
        if !condition() { fail(message(), file: file, line: line) }
    }
    static func expect(_ condition: @autoclosure () -> Bool, _ message:String, file: StaticString = #file, line: UInt = #line) {
        check(condition(),message,file:file,line:line)
    }
    static func main() throws {
        let original = MLMultiArray(MLShapedArray<Float>(scalars:[1,-2,0,4],shape:[1,2,2]))
        let reused = try SongFormerTensorBridge.float32(original)
        expect(original === reused,"Contiguous Float32 must be reused")

        // Exhaustive finite binary16 patterns, including signed zero and subnormals.
        let bits = (UInt32(0)...UInt32(65535)).map(UInt16.init).filter { $0 & 0x7c00 != 0x7c00 }
        let half = try MLMultiArray(shape:[1,NSNumber(value:bits.count)],dataType:.float16)
        let halfBuffer = half.dataPointer.assumingMemoryBound(to:UInt16.self)
        for (i,value) in bits.enumerated() { halfBuffer[i] = value }
        // Exact widening is the reference: every finite binary16 value is representable in binary32.
        let reference = bits.map { Float(Float16(bitPattern:$0)) }
        let converted = try SongFormerTensorBridge.float32(half)
        let values = MLShapedArray<Float>(converted).scalars
        let mismatches = zip(values,reference).filter { $0.0.bitPattern != $0.1.bitPattern }.count
        expect(mismatches == 0,"Float16 conversion must be exact for all finite binary16 values; \(mismatches) differ")
        // Informative only: the system converter's behaviour on subnormals/signed zero can change by OS.
        let system = MLShapedArray<Float>(converting:half).scalars
        let systemDifferences = zip(system,reference).filter { $0.0.bitPattern != $0.1.bitPattern }
        print("Core ML Float16 converter differs from exact widening on \(systemDifferences.count) of \(bits.count) finite values" +
              (systemDifferences.isEmpty ? "" : " (e.g. exact \(systemDifferences[0].1) vs system \(systemDifferences[0].0))"))
        // A strided Float16 view must convert in logical order and exactly.
        let halfStorage = UnsafeMutablePointer<Float16>.allocate(capacity:8)
        halfStorage.initialize(repeating:99,count:8)
        halfStorage[0] = Float16(bitPattern:0x0001); halfStorage[1] = 1.5; halfStorage[4] = Float16(bitPattern:0x8000); halfStorage[5] = -2
        let halfView = try MLMultiArray(dataPointer:halfStorage,shape:[1,2,2],dataType:.float16,strides:[8,4,1],deallocator: { raw in
            raw.assumingMemoryBound(to:Float16.self).deinitialize(count:8)
            raw.deallocate()
        })
        let halfStrided = MLShapedArray<Float>(try SongFormerTensorBridge.float32(halfView)).scalars
        expect(halfStrided.map(\.bitPattern) == [Float(Float16(bitPattern:0x0001)),1.5,-0.0,-2].map(\.bitPattern),"Strided Float16 order/exactness")

        // A strided view with padding must retain logical order, not copy padding as data.
        let storage = UnsafeMutablePointer<Float>.allocate(capacity:8)
        storage.initialize(repeating:99,count:8)
        storage[0]=1;storage[1]=2;storage[4]=3;storage[5]=4
        let view = try MLMultiArray(dataPointer:storage,shape:[1,2,2],dataType:.float32,strides:[8,4,1],deallocator: { raw in
            raw.assumingMemoryBound(to:Float.self).deinitialize(count:8)
            raw.deallocate()
        })
        let canonical = try SongFormerTensorBridge.float32(view)
        expect(MLShapedArray<Float>(canonical).scalars == [1,2,3,4],"Strided fallback order")

        for frames in [1,3,750,3000] {
            let width = frames < 10 ? 2 : 1024
            let streams = (0..<4).map { stream in (0..<(frames*width)).map { Float(stream*100000+$0) } }
            let fused = try SongFormerTensorBridge.fuse(streams:streams,frames:frames,width:width)
            let result = MLShapedArray<Float>(fused).scalars
            for frame in 0..<frames {
                for stream in 0..<4 {
                    for channel in 0..<width {
                        expect(result[frame*4*width+stream*width+channel] == streams[stream][frame*width+channel],"Fusion changed stream/frame/channel ordering")
                    }
                }
            }
        }
        // MuQ-only: two streams fuse to [1, frames, 2*width] in the given order.
        let pairStreams = (0..<2).map { stream in (0..<(6*3)).map { Float(stream*1000+$0) } }
        let pair = try SongFormerTensorBridge.fuse(streams:pairStreams,frames:6,width:3)
        expect(pair.shape.map(\.intValue) == [1,6,6],"Two-stream fused shape")
        let pairValues = MLShapedArray<Float>(pair).scalars
        for frame in 0..<6 { for stream in 0..<2 { for channel in 0..<3 {
            expect(pairValues[frame*6+stream*3+channel] == pairStreams[stream][frame*3+channel],"Two-stream fusion order")
        } } }
        for count in [1,3] {
            do { _ = try SongFormerTensorBridge.fuse(streams:Array(repeating:[Float](repeating:0,count:4),count:count),frames:2,width:2);fail("Stream count \(count) accepted") }
            catch SongFormerTensorBridge.BridgeError.invalid { }
        }
        let bad = MLMultiArray(MLShapedArray<Float>(scalars:[Float.nan],shape:[1,1]))
        do { _ = try SongFormerTensorBridge.float32(bad); fail("NaN should be rejected") }
        catch SongFormerTensorBridge.BridgeError.invalid { }
        let inf = try MLMultiArray(shape:[1,1],dataType:.float16)
        inf.dataPointer.assumingMemoryBound(to:UInt16.self)[0] = 0x7c00
        do { _ = try SongFormerTensorBridge.float32(inf); fail("FP16 Inf should be rejected") }
        catch SongFormerTensorBridge.BridgeError.invalid { }
        do { _ = try SongFormerTensorBridge.fuse(streams:[[1],[2],[3],[4]],frames:2,width:1);fail("Short streams should be rejected") }
        catch SongFormerTensorBridge.BridgeError.invalid { }
        print("Tensor bridge checks passed: exact binary16 widening (contiguous and strided), Float32 reuse, strided layout, fused stream order, invalid input rejection.")
    }
}
