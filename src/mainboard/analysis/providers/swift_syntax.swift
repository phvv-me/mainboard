import Foundation
import SwiftDiagnostics
import SwiftParser
import SwiftParserDiagnostics
import SwiftSyntax

struct Input: Decodable {
  let root: String
  let files: [String]
}
struct Owner {
  let id: Int
  let name: String
  let kind: String
  let inherited: [String]
}

final class Declarations: SyntaxVisitor {
  var names = Set<String>()
  init() { super.init(viewMode: .sourceAccurate) }
  override func visit(_ node: StructDeclSyntax) -> SyntaxVisitorContinueKind {
    names.insert(node.name.text)
    return .visitChildren
  }
  override func visit(_ node: ClassDeclSyntax) -> SyntaxVisitorContinueKind {
    names.insert(node.name.text)
    return .visitChildren
  }
  override func visit(_ node: EnumDeclSyntax) -> SyntaxVisitorContinueKind {
    names.insert(node.name.text)
    return .visitChildren
  }
  override func visit(_ node: ActorDeclSyntax) -> SyntaxVisitorContinueKind {
    names.insert(node.name.text)
    return .visitChildren
  }
  override func visit(_ node: ProtocolDeclSyntax) -> SyntaxVisitorContinueKind {
    names.insert(node.name.text)
    return .visitChildren
  }
  override func visit(_ node: TypeAliasDeclSyntax) -> SyntaxVisitorContinueKind {
    names.insert(node.name.text)
    return .visitChildren
  }
  override func visit(_ node: FunctionDeclSyntax) -> SyntaxVisitorContinueKind {
    names.insert(node.name.text)
    return .visitChildren
  }
  override func visit(_ node: FunctionParameterSyntax) -> SyntaxVisitorContinueKind {
    names.insert((node.secondName ?? node.firstName).text)
    return .visitChildren
  }
  override func visit(_ node: GenericParameterSyntax) -> SyntaxVisitorContinueKind {
    names.insert(node.name.text)
    return .visitChildren
  }
  override func visit(_ node: VariableDeclSyntax) -> SyntaxVisitorContinueKind {
    for binding in node.bindings {
      if let name = binding.pattern.as(IdentifierPatternSyntax.self) {
        names.insert(name.identifier.text)
      }
    }
    return .visitChildren
  }
}

final class Imports: SyntaxVisitor {
  var names = Set<String>()
  init() { super.init(viewMode: .sourceAccurate) }
  override func visit(_ node: ImportDeclSyntax) -> SyntaxVisitorContinueKind {
    names.insert(node.path.map { $0.name.text }.joined(separator: "."))
    return .skipChildren
  }
}

final class FactsVisitor: SyntaxVisitor {
  let path: String
  let converter: SourceLocationConverter
  let imports: Set<String>
  let declarations: Set<String>
  var tables: [String: [[Any]]] = [:]
  init(path: String, tree: SourceFileSyntax, imports: Set<String>, declarations: Set<String>) {
    self.path = path
    converter = SourceLocationConverter(fileName: path, tree: tree)
    self.imports = imports
    self.declarations = declarations
    super.init(viewMode: .sourceAccurate)
  }
  func placement(_ node: some SyntaxProtocol) -> [Any] {
    let location = converter.location(for: node.positionAfterSkippingLeadingTrivia)
    return ["swift:\(path)", path, "swift", location.line, location.column - 1]
  }
  func append(_ table: String, _ node: some SyntaxProtocol, _ values: [Any]) {
    tables[table, default: []].append(placement(node) + values)
  }
  func shadowed(_ name: String) -> Int {
    (declarations.contains(String(name.split(separator: ".").first ?? "")) ? 1 : 0)
  }
  func owner(_ syntax: Syntax?) -> Owner? {
    guard let syntax else { return nil }
    func make(_ name: String, _ kind: String, _ clause: InheritanceClauseSyntax?) -> Owner {
      Owner(
        id: syntax.positionAfterSkippingLeadingTrivia.utf8Offset, name: name, kind: kind,
        inherited: clause?.inheritedTypes.map { $0.type.trimmedDescription } ?? [])
    }
    if let node = syntax.as(StructDeclSyntax.self) {
      return make(node.name.text, "struct", node.inheritanceClause)
    }
    if let node = syntax.as(ClassDeclSyntax.self) {
      return make(node.name.text, "class", node.inheritanceClause)
    }
    if let node = syntax.as(EnumDeclSyntax.self) {
      return make(node.name.text, "enum", node.inheritanceClause)
    }
    if let node = syntax.as(ActorDeclSyntax.self) {
      return make(node.name.text, "actor", node.inheritanceClause)
    }
    if let node = syntax.as(ExtensionDeclSyntax.self) {
      return make(node.extendedType.trimmedDescription, "extension", node.inheritanceClause)
    }
    return nil
  }
  func container(_ node: some SyntaxProtocol) -> Owner? {
    var parent = node.parent
    while let current = parent {
      if let owner = owner(current) { return owner }
      if current.is(FunctionDeclSyntax.self) || current.is(ClosureExprSyntax.self) { return nil }
      parent = current.parent
    }
    return nil
  }
  func record(
    _ syntax: some SyntaxProtocol, _ name: String, _ kind: String,
    _ clause: InheritanceClauseSyntax?
  ) {
    for inherited in clause?.inheritedTypes ?? [] {
      let type = inherited.type.trimmedDescription
      append(
        "conformances", inherited,
        [syntax.positionAfterSkippingLeadingTrivia.utf8Offset, name, kind, type, shadowed(type)])
    }
  }
  override func visit(_ node: ImportDeclSyntax) -> SyntaxVisitorContinueKind {
    append("imports", node, [node.path.map { $0.name.text }.joined(separator: ".")])
    return .skipChildren
  }
  override func visit(_ node: StructDeclSyntax) -> SyntaxVisitorContinueKind {
    record(node, node.name.text, "struct", node.inheritanceClause)
    return .visitChildren
  }
  override func visit(_ node: ClassDeclSyntax) -> SyntaxVisitorContinueKind {
    record(node, node.name.text, "class", node.inheritanceClause)
    return .visitChildren
  }
  override func visit(_ node: EnumDeclSyntax) -> SyntaxVisitorContinueKind {
    record(node, node.name.text, "enum", node.inheritanceClause)
    return .visitChildren
  }
  override func visit(_ node: ActorDeclSyntax) -> SyntaxVisitorContinueKind {
    record(node, node.name.text, "actor", node.inheritanceClause)
    return .visitChildren
  }
  override func visit(_ node: ExtensionDeclSyntax) -> SyntaxVisitorContinueKind {
    record(node, node.extendedType.trimmedDescription, "extension", node.inheritanceClause)
    return .visitChildren
  }
  override func visit(_ node: FunctionCallExprSyntax) -> SyntaxVisitorContinueKind {
    if let target = node.calledExpression.as(DeclReferenceExprSyntax.self) {
      append("calls", node, [target.baseName.text, "", shadowed(target.baseName.text)])
    } else if let member = node.calledExpression.as(MemberAccessExprSyntax.self),
      let base = member.base
    {
      append(
        "calls", node,
        [
          member.declName.baseName.text, base.trimmedDescription,
          shadowed(base.trimmedDescription + "." + member.declName.baseName.text),
        ])
    }
    return .visitChildren
  }
  override func visit(_ node: DeclReferenceExprSyntax) -> SyntaxVisitorContinueKind {
    if let member = node.parent?.as(MemberAccessExprSyntax.self), member.declName.id == node.id {
      return .visitChildren
    }
    append("references", node, [node.baseName.text, "", shadowed(node.baseName.text)])
    return .visitChildren
  }
  override func visit(_ node: IdentifierTypeSyntax) -> SyntaxVisitorContinueKind {
    append("references", node, [node.name.text, "", shadowed(node.name.text)])
    return .visitChildren
  }
  override func visit(_ node: MemberTypeSyntax) -> SyntaxVisitorContinueKind {
    append(
      "references", node,
      [
        node.name.text, node.baseType.trimmedDescription,
        shadowed(node.baseType.trimmedDescription + "." + node.name.text),
      ])
    return .visitChildren
  }
  override func visit(_ node: MemberAccessExprSyntax) -> SyntaxVisitorContinueKind {
    if let base = node.base {
      append(
        "references", node,
        [
          node.declName.baseName.text, base.trimmedDescription,
          shadowed(base.trimmedDescription + "." + node.declName.baseName.text),
        ])
    }
    return .visitChildren
  }
  override func visit(_ node: AttributeSyntax) -> SyntaxVisitorContinueKind {
    let name = node.attributeName.trimmedDescription
    let enclosing = container(node)
    var declaration = node.parent
    while let current = declaration, !current.is(DeclSyntax.self) { declaration = current.parent }
    let ownerID = declaration?.positionAfterSkippingLeadingTrivia.utf8Offset ?? 0
    var ownerName = "declaration"
    var isStatic = 0
    if let node = declaration?.as(VariableDeclSyntax.self) {
      ownerName = node.bindings.first?.pattern.trimmedDescription ?? "property"
      isStatic = node.modifiers.contains { ["static", "class"].contains($0.name.text) } ? 1 : 0
    } else if let node = declaration?.as(FunctionDeclSyntax.self) {
      ownerName = node.name.text
    } else if let own = owner(declaration) {
      ownerName = own.name
    }
    append(
      "attributes", node,
      [
        ownerID, ownerName, name, enclosing?.id ?? -1, enclosing?.name ?? "", enclosing?.kind ?? "",
        isStatic, shadowed(name),
      ])
    return .visitChildren
  }
  override func visit(_ node: VariableDeclSyntax) -> SyntaxVisitorContinueKind {
    guard let owner = container(node), imports.contains("SwiftUI"),
      owner.inherited.contains(where: {
        $0 == "SwiftUI.View" || ($0 == "View" && !declarations.contains("View"))
      })
    else { return .visitChildren }
    for element in node.attributes {
      guard let attribute = element.as(AttributeSyntax.self) else { continue }
      let name = attribute.attributeName.trimmedDescription
      guard ["State", "StateObject", "SwiftUI.State", "SwiftUI.StateObject"].contains(name),
        shadowed(name) == 0
      else { continue }
      let isPrivate = node.modifiers.contains { $0.name.text == "private" && $0.detail == nil }
      for binding in node.bindings {
        append("states", attribute, [binding.pattern.trimmedDescription, name, isPrivate ? 1 : 0])
      }
    }
    for binding in node.bindings {
      guard binding.pattern.as(IdentifierPatternSyntax.self)?.identifier.text == "body",
        let type = binding.typeAnnotation?.type.as(SomeOrAnyTypeSyntax.self),
        type.someOrAnySpecifier.text == "some",
        ["View", "SwiftUI.View"].contains(type.constraint.trimmedDescription),
        let accessors = binding.accessorBlock, case .getter(let statements) = accessors.accessors,
        statements.count == 1, let statement = statements.first,
        let call = statement.item.as(FunctionCallExprSyntax.self),
        let target = call.calledExpression.as(DeclReferenceExprSyntax.self),
        target.baseName.text == "AnyView",
        !declarations.contains("AnyView")
      else { continue }
      append("erasures", call, [owner.name])
    }
    return .visitChildren
  }
}

func main() throws {
  let input = try JSONDecoder().decode(
    Input.self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
  var trees: [(String, SourceFileSyntax, [Diagnostic])] = []
  let declarations = Declarations()
  for path in input.files {
    let source = try String(
      contentsOfFile: URL(fileURLWithPath: input.root).appendingPathComponent(path).path,
      encoding: .utf8)
    let tree = Parser.parse(source: source)
    let errors = ParseDiagnosticsGenerator.diagnostics(for: tree).filter {
      $0.diagMessage.severity == .error
    }
    trees.append((path, tree, errors))
    declarations.walk(tree)
  }
  var output: [String: [[Any]]] = [:]
  for (path, tree, errors) in trees {
    let importVisitor = Imports()
    importVisitor.walk(tree)
    let imports = importVisitor.names
    output["facts", default: []].append([
      "swift:\(path)", path, "swift", 1, 0, errors.count,
      errors.map { $0.message }.joined(separator: "; "),
    ])
    guard errors.isEmpty else { continue }
    let visitor = FactsVisitor(
      path: path, tree: tree, imports: imports, declarations: declarations.names)
    visitor.walk(tree)
    for (table, rows) in visitor.tables { output[table, default: []].append(contentsOf: rows) }
  }
  let data = try JSONSerialization.data(withJSONObject: output, options: [.sortedKeys])
  FileHandle.standardOutput.write(data)
}
do { try main() } catch {
  FileHandle.standardError.write(Data("SwiftSyntax: \(error)\n".utf8))
  exit(1)
}
