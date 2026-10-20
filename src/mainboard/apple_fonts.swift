import CoreText
import Foundation
import ImageIO
import UniformTypeIdentifiers

struct FontRequest: Decodable {
  let fonts: [String]
  let text: String
  let output: String
  let width: Double
  let size: Double
  let scale: Double
}
struct FontPreview: Encodable {
  let source: String
  let image: String
  let font: String
  let width: Int
  let height: Int
}
struct FontPreviews: Encodable { let images: [FontPreview] }
struct FontFailure: LocalizedError {
  let message: String
  var errorDescription: String? { message }
}
func render(_ path: String, request: FontRequest) throws -> (FontPreview, Data) {
  let source = URL(fileURLWithPath: path)
  guard
    let descriptors = CTFontManagerCreateFontDescriptorsFromURL(source as CFURL)
      as? [CTFontDescriptor], let descriptor = descriptors.first,
    let expectedName = CTFontDescriptorCopyAttribute(descriptor, kCTFontNameAttribute) as? String
  else { throw FontFailure(message: "Cannot read font: \(source.lastPathComponent)") }
  let font = CTFontCreateWithFontDescriptor(descriptor, request.size, nil)
  let name = CTFontCopyPostScriptName(font) as String
  guard name == expectedName else {
    throw FontFailure(
      message: "Core Text substituted a different font for \(source.lastPathComponent)")
  }
  let characters = Array(String(request.text.filter { !$0.isWhitespace }).utf16)
  var glyphs = [CGGlyph](repeating: 0, count: characters.count)
  guard CTFontGetGlyphsForCharacters(font, characters, &glyphs, characters.count) else {
    throw FontFailure(message: "Font lacks preview glyphs: \(source.lastPathComponent)")
  }
  let text = NSAttributedString(
    string: request.text,
    attributes: [
      NSAttributedString.Key(kCTFontAttributeName as String): font,
      NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(gray: 0, alpha: 1),
    ])
  let setter = CTFramesetterCreateWithAttributedString(text)
  var fit = CFRange()
  let measured = CTFramesetterSuggestFrameSizeWithConstraints(
    setter, CFRange(), nil, CGSize(width: request.width, height: .greatestFiniteMagnitude), &fit)
  let height = ceil(measured.height + 2)
  let pixelWidth = ceil(request.width * request.scale)
  let pixelHeight = ceil(height * request.scale)
  guard fit.length == text.length, height.isFinite, pixelWidth.isFinite, pixelHeight.isFinite,
    pixelWidth > 0, pixelHeight > 0, pixelWidth * pixelHeight <= 64_000_000,
    let space = CGColorSpace(name: CGColorSpace.sRGB),
    let context = CGContext(
      data: nil, width: Int(pixelWidth), height: Int(pixelHeight), bitsPerComponent: 8,
      bytesPerRow: 0, space: space,
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    )
  else { throw FontFailure(message: "Preview does not fit the requested canvas") }
  context.clear(CGRect(x: 0, y: 0, width: pixelWidth, height: pixelHeight))
  context.scaleBy(x: request.scale, y: request.scale)
  context.textMatrix = .identity
  let frame = CTFramesetterCreateFrame(
    setter, CFRange(),
    CGPath(rect: CGRect(x: 0, y: 0, width: request.width, height: height), transform: nil), nil)
  CTFrameDraw(frame, context)
  let data = NSMutableData()
  guard let image = context.makeImage(),
    let destination = CGImageDestinationCreateWithData(
      data, UTType.png.identifier as CFString, 1, nil)
  else { throw FontFailure(message: "Cannot encode font preview") }
  CGImageDestinationAddImage(destination, image, nil)
  guard CGImageDestinationFinalize(destination) else {
    throw FontFailure(message: "Cannot finish font preview PNG")
  }
  let output = URL(fileURLWithPath: request.output)
    .appendingPathComponent(source.deletingPathExtension().lastPathComponent + ".png")
  return (
    FontPreview(
      source: path, image: output.path, font: name,
      width: Int(pixelWidth), height: Int(pixelHeight)), data as Data
  )
}
do {
  guard CommandLine.arguments.count == 2 else {
    throw FontFailure(message: "Expected font render request")
  }
  let request = try JSONDecoder().decode(
    FontRequest.self,
    from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
  guard !request.fonts.isEmpty,
    !request.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
    request.text.utf16.count <= 16384,
    [request.width, request.size, request.scale].allSatisfy({ $0.isFinite && $0 > 0 }),
    request.width <= 8192, request.size <= 1024, request.scale <= 8
  else { throw FontFailure(message: "Invalid font preview dimensions or text") }
  let folder = URL(fileURLWithPath: request.output)
  guard !FileManager.default.fileExists(atPath: folder.path) else {
    throw FontFailure(message: "Refusing to overwrite font preview output")
  }
  let rendered = try request.fonts.map { try render($0, request: request) }
  try FileManager.default.createDirectory(
    at: folder.deletingLastPathComponent(), withIntermediateDirectories: true)
  try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
  for (preview, bytes) in rendered {
    try bytes.write(to: URL(fileURLWithPath: preview.image), options: .withoutOverwriting)
  }
  let encoder = JSONEncoder()
  encoder.outputFormatting = .sortedKeys
  print(
    String(decoding: try encoder.encode(FontPreviews(images: rendered.map { $0.0 })), as: UTF8.self)
  )
} catch {
  FileHandle.standardError.write(Data((error.localizedDescription + "\n").utf8))
  exit(1)
}
