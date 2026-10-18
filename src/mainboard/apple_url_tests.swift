import Darwin
import Foundation

struct WarmLinkRequest: Decodable {
  let id: UUID
  let url: URL
}

struct CommandFailure: Error { let message: String }
func xcrun(_ arguments: [String]) throws -> String {
  let process = Process()
  let output = Pipe()
  process.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
  process.arguments = arguments
  process.standardOutput = output
  process.standardError = FileHandle.nullDevice
  try process.run()
  let data = output.fileHandleForReading.readDataToEndOfFile()
  process.waitUntilExit()
  guard process.terminationStatus == 0 else {
    throw CommandFailure(message: arguments.joined(separator: " "))
  }
  return String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
}

struct URLTestConfiguration: Decodable {
  let device: String
  let runner_bundle: String
  let marker: String
  let routes: [String]
  let command: [String]
}
guard CommandLine.arguments.count == 2 else {
  fatalError("Expected URL test execution descriptor")
}
let configuration = try JSONDecoder().decode(
  URLTestConfiguration.self,
  from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
let device = configuration.device
let test = Process()
test.executableURL = URL(fileURLWithPath: configuration.command[0])
test.arguments = Array(configuration.command.dropFirst())
test.standardOutput = FileHandle.standardOutput
test.standardError = FileHandle.standardError
let started = Date()
var consumed = Set<UUID>()
var delivered = Set<String>()
var marker: URL?
var nextContainerCheck = Date.distantPast
let allowed = Set(configuration.routes.compactMap(URL.init(string:)).map(\.absoluteString))
try test.run()
do {
  while test.isRunning {
    if Date() >= nextContainerCheck {
      nextContainerCheck = Date().addingTimeInterval(1)
      if let container = try? xcrun([
        "simctl", "get_app_container", device,
        configuration.runner_bundle, "data",
      ]) {
        marker = URL(fileURLWithPath: container).appendingPathComponent(configuration.marker)
      }
    }
    if let marker,
      let attributes = try? FileManager.default.attributesOfItem(atPath: marker.path),
      let modified = attributes[.modificationDate] as? Date, modified >= started,
      let bytes = try? Data(contentsOf: marker),
      let request = try? JSONDecoder().decode(WarmLinkRequest.self, from: bytes),
      consumed.insert(request.id).inserted
    {
      guard allowed.contains(request.url.absoluteString)
      else { throw CommandFailure(message: "Unexpected test route") }
      _ = try xcrun(["simctl", "openurl", device, request.url.absoluteString])
      delivered.insert(request.url.absoluteString)
      FileHandle.standardError.write(
        Data("Delivered warm study route at \(Date()): \(request.url.path)\n".utf8))
    }
    Thread.sleep(forTimeInterval: 0.1)
  }
  test.waitUntilExit()
  guard test.terminationStatus == 0 else { exit(test.terminationStatus) }
  guard delivered == allowed else {
    throw CommandFailure(message: "Not all warm routes were delivered")
  }
} catch {
  if test.isRunning {
    test.terminate()
    test.waitUntilExit()
  }
  throw error
}
