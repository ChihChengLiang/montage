import Foundation
import Vision
import CoreImage
// usage: vis img.jpg  -> JSON {text:[[x,y,w,h]...], attn:[[x,y,w,h]], obj:[[x,y,w,h]]}  (normalized, origin bottom-left)
let path = CommandLine.arguments[1]
let url = URL(fileURLWithPath: path)
let handler = VNImageRequestHandler(url: url, options: [:])
let tr = VNRecognizeTextRequest(); tr.recognitionLevel = .accurate; tr.usesLanguageCorrection = false
tr.recognitionLanguages = ["en-US","zh-Hant"]
let ar = VNGenerateAttentionBasedSaliencyImageRequest()
let or_ = VNGenerateObjectnessBasedSaliencyImageRequest()
try handler.perform([tr, ar, or_])
func r(_ b: CGRect) -> [Double] { [Double(b.minX), Double(b.minY), Double(b.width), Double(b.height)] }
var out: [String: Any] = [:]
out["text"] = (tr.results ?? []).map { r($0.boundingBox) }
out["textstr"] = (tr.results ?? []).map { $0.topCandidates(1).first?.string ?? "" }
out["attn"] = (ar.results?.first?.salientObjects ?? []).map { r($0.boundingBox) }
out["obj"] = (or_.results?.first?.salientObjects ?? []).map { r($0.boundingBox) }
// saliency heatmap -> save small png of attention map as JSON grid
func grid(_ pb: CVPixelBuffer?) -> [[Float]] {
  guard let pb = pb else { return [] }
  CVPixelBufferLockBaseAddress(pb, .readOnly); defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
  let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
  let bpr = CVPixelBufferGetBytesPerRow(pb); let base = CVPixelBufferGetBaseAddress(pb)!
  var g = [[Float]]()
  for y in 0..<h { let row = base.advanced(by: y*bpr).assumingMemoryBound(to: Float.self); g.append((0..<w).map{ row[$0] }) }
  return g
}
out["attn_map"] = grid(ar.results?.first?.pixelBuffer)
out["obj_map"] = grid(or_.results?.first?.pixelBuffer)
let d = try JSONSerialization.data(withJSONObject: out)
FileHandle.standardOutput.write(d)
