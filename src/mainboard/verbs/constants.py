"""The whole command line, declared once without importing a verb module.

`TREE` feeds registration, aliases, help, the unknown-command answer, the listing and the tree
view: `mb --help` reads it alone, and a group's module is imported only when that group is typed.
Regrouping the CLI is an edit of this table and of the bindings ending the home modules it names.
A node holds its commands, then its groups, in the order `mb` lists them.
"""

from .enums import Accent, Scope
from .node import Command, Node

TREE = Node(
    name="mb",
    help="Environments, dispatch and evidence in one tool.",
    # Each option's help is the docstring of the `Root` method it runs.
    options=(
        Command(name="--help", aliases=("help",)),
        Command(name="--version", moved=("self version",)),
        Command(name="--tree"),
        Command(name="--completion", moved=("completion",)),
    ),
    commands=(
        Command(
            name="attest",
            module="mainboard.verbs.work.running",
            help="Record what this machine looks like right now into a stream's receipts, once.",
            show=False,
        ),
        Command(
            name="execute",
            module="mainboard.verbs.work.running",
            help=(
                "Run a dispatched job from its record, which every generated job script hands "
                "over."
            ),
            show=False,
        ),
        Command(
            name="provide",
            module="mainboard.verbs.work.running",
            help=(
                "Build the immutable environment a dispatched job activates, and print where it "
                "is."
            ),
            show=False,
        ),
        Command(
            name="sample",
            module="mainboard.verbs.work.running",
            help=(
                "Publish this machine's live readings into a stream's receipts until told to stop."
            ),
            show=False,
        ),
    ),
    children=(
        Node(
            name="run",
            module="mainboard.verbs.work.running",
            help="Run a command or a file::job through the host's plan.",
            commands=(Command(name="run", default=True, trailing=True),),
        ),
        Node(
            name="dep",
            module="mainboard.verbs.workspace.environments",
            help="Declare, solve and install environments; pixi's verbs mean what pixi's mean.",
            accent=Accent.GREEN,
            commands=(
                Command(name="install", aliases=("install",)),
                Command(name="add"),
                Command(name="remove"),
                Command(name="update", aliases=("update",)),
                Command(name="upgrade", aliases=("upgrade",)),
                Command(name="lock"),
                Command(name="list", moved=("list",), trailing=True),
                Command(name="tree", aliases=("tree",), trailing=True),
                Command(
                    name="shell",
                    module="mainboard.verbs.work.running",
                    help=(
                        "Open an interactive shell in this workspace's environment, here or on a "
                        "host."
                    ),
                    trailing=True,
                ),
            ),
            children=(
                Node(
                    name="binary",
                    module="mainboard.verbs.workspace.environments",
                    help=(
                        "Install and find the executables a manifest pins by release archive and "
                        "hash."
                    ),
                    commands=(
                        Command(name="sync", default=True),
                        Command(name="list", scope=Scope.READ),
                        Command(name="path", scope=Scope.READ),
                    ),
                ),
            ),
        ),
        Node(
            name="job",
            module="mainboard.verbs.work.submit",
            help="Dispatch jobs to hosts and follow them until they settle.",
            accent=Accent.BLUE,
            commands=(
                Command(name="submit", trailing=True),
                Command(name="list"),
                Command(name="show"),
                Command(name="logs"),
                Command(name="cancel"),
                Command(name="digest"),
                Command(name="events", show=False),
            ),
            children=(
                Node(
                    name="monitor",
                    module="mainboard.verbs.work.monitoring",
                    help="Start, stop and inspect the periodic pass that settles jobs unattended.",
                    commands=(
                        Command(name="monitor", default=True, show=False),
                        Command(name="start"),
                        Command(name="status", scope=Scope.READ),
                        Command(name="stop"),
                    ),
                ),
            ),
        ),
        Node(
            name="host",
            module="mainboard.verbs.machines.hosts",
            help="Set up, list, inspect and reach the machines jobs run on.",
            accent=Accent.BLUE,
            commands=(
                Command(name="list"),
                Command(name="setup"),
                Command(name="sync"),
                Command(
                    name="evict",
                    module="mainboard.verbs.evidence.lake",
                    help="Remove a host's copies of evidence the center's lake holds, to give its "
                    "disk back.",
                ),
                Command(name="update", moved=("self update",)),
                Command(name="upgrade"),
                Command(
                    name="pack",
                    module="mainboard.verbs.workspace.environments",
                    help=(
                        "Build an environment into files a machine runs without installing "
                        "anything."
                    ),
                ),
                Command(name="offers"),
                Command(name="hold"),
                Command(name="release"),
                Command(name="serve", show=False),
            ),
        ),
        Node(
            name="proc",
            module="mainboard.verbs.machines.procs",
            help="Kill a process tree, bound a command, wait for a file or port, on every system.",
            accent=Accent.BLUE,
            commands=(
                Command(name="timeout", trailing=True),
                Command(name="wait"),
                Command(name="kill"),
                Command(name="list"),
            ),
        ),
        Node(
            name="code",
            module="mainboard.verbs.workspace.core",
            help="Check, fix and move the code; build its apps and its agents' configuration.",
            accent=Accent.MAGENTA,
            commands=(
                Command(name="lint"),
                Command(name="ci"),
                Command(name="doctor", aliases=("doctor",)),
                Command(name="new"),
            ),
            children=(
                Node(
                    name="rules",
                    module="mainboard.verbs.apps.refactor",
                    help="Judge a project by the rules and fix it; explain, test and query them.",
                    commands=(
                        Command(name="check", moved=("refactor check",)),
                        Command(name="fix", moved=("refactor fix",)),
                        Command(name="preview", moved=("refactor preview",)),
                        Command(name="trial", moved=("refactor trial",)),
                        Command(name="gate", moved=("refactor gate",)),
                        Command(name="explain", moved=("refactor explain", "refactor catalog")),
                        Command(name="test", moved=("refactor test",)),
                        Command(name="sql", moved=("refactor sql",)),
                    ),
                ),
                Node(
                    name="refactor",
                    module="mainboard.verbs.apps.refactor",
                    help="Move and prove code, and map its uses, impact and Swift inventory.",
                    commands=(
                        Command(name="move"),
                        Command(name="proof"),
                        Command(name="uses"),
                        Command(name="impact"),
                        Command(name="diagram"),
                        Command(name="matrix"),
                        Command(name="swift-inventory"),
                        Command(name="swift-compare"),
                    ),
                ),
                Node(
                    name="app",
                    module="mainboard.verbs.apps.core",
                    help="Build, test and launch the apps a manifest declares, and their Xcode.",
                    commands=(
                        Command(name="list"),
                        Command(name="build"),
                        Command(name="test"),
                        Command(name="launch", moved=("apple launch",)),
                        Command(name="execute", moved=("apple execute",)),
                        Command(
                            name="inspect",
                            moved=("apple bundle", "apple diagnostics", "apple results"),
                        ),
                        Command(name="simulators", moved=("apple simulators",)),
                        Command(name="toolchain", moved=("apple toolchain",)),
                        Command(name="fonts", moved=("apple font-previews",)),
                    ),
                ),
                Node(
                    name="agents",
                    module="mainboard.verbs.workspace.agents",
                    help=(
                        "Configure every coding agent alike from `.agents`: files, links, "
                        "servers, releases."
                    ),
                    aliases=("agents",),
                    commands=(Command(name="check"), Command(name="sync"), Command(name="update")),
                ),
            ),
        ),
        Node(
            name="git",
            module="mainboard.verbs.workspace.git",
            help="Operate the workspace repository and its owned submodules as one tree.",
            accent=Accent.MAGENTA,
            commands=(
                Command(name="status"),
                Command(name="commit"),
                Command(name="push"),
                Command(name="pull"),
                Command(name="check"),
            ),
        ),
        Node(
            name="data",
            module="mainboard.verbs.workspace.core",
            help="Query, keep and publish the workspace's evidence: results, lake, papers.",
            accent=Accent.CYAN,
            commands=(
                Command(name="query", aliases=("query",)),
                Command(
                    name="probe",
                    module="mainboard.verbs.evidence.papers",
                    help="Inspect CUDA device code and append its evidence to the workspace lake.",
                    moved=("probe device",),
                ),
            ),
            children=(
                Node(
                    name="lake",
                    module="mainboard.verbs.evidence.lake",
                    help="Keep the workspace's state lake and the evidence it holds.",
                    commands=(
                        Command(name="ingest", aliases=("lake ingest",)),
                        Command(name="materialize"),
                        Command(
                            name="collect",
                            module="mainboard.verbs.work.submit",
                            help=(
                                "Collect remote evidence for queries, including runs started "
                                "directly on that node."
                            ),
                        ),
                        Command(name="check"),
                        Command(name="compact"),
                        Command(name="dedup"),
                        Command(name="evict"),
                        Command(name="replicate"),
                        Command(name="upgrade"),
                        Command(name="serve", show=False),
                    ),
                ),
                Node(
                    name="paper",
                    module="mainboard.verbs.evidence.papers",
                    help="Build a declared manuscript, plot its figures, audit what it rests on.",
                    commands=(
                        Command(name="build", aliases=("paper build",)),
                        Command(
                            name="plot",
                            module="mainboard.verbs.evidence.plotting",
                            help=(
                                "Plot a local SELECT using Seaborn and Matplotlib, without "
                                "implicit aggregation."
                            ),
                            aliases=("plot",),
                        ),
                        Command(name="evidence", moved=("refactor evidence",)),
                        Command(name="prose", moved=("refactor prose",)),
                    ),
                ),
            ),
        ),
        Node(
            name="memory",
            module="mainboard.verbs.evidence.memory",
            help="Find and keep what AIZK remembers; a keep made while it is down waits.",
            accent=Accent.CYAN,
            commands=(
                Command(name="find", scope=Scope.READ),
                Command(name="keep", scope=Scope.AGENT),
                Command(name="status", scope=Scope.READ),
            ),
        ),
        Node(
            name="serve",
            module="mainboard.verbs.serve.core",
            help="Start every service; one supervisor this machine keeps alive runs them.",
            accent=Accent.BLUE,
            commands=(
                Command(name="serve", default=True, scope=Scope.OWNER),
                Command(name="status", scope=Scope.OWNER),
                Command(name="stop", scope=Scope.OWNER),
                Command(
                    name="sessions",
                    module="mainboard.verbs.serve.telemetry",
                    help="Show the agent sessions of the last days, or one session's turns.",
                    scope=Scope.READ,
                ),
                Command(
                    name="usage",
                    module="mainboard.verbs.serve.telemetry",
                    help="Show what the agents used: the last 5 hours and 7 days, then each day.",
                    scope=Scope.READ,
                ),
                Command(
                    name="asks",
                    module="mainboard.verbs.serve.asks",
                    help="Show the asks waiting for the owner's answer, oldest first.",
                    scope=Scope.READ,
                ),
                Command(
                    name="answer",
                    module="mainboard.verbs.serve.asks",
                    help=(
                        "Answer one waiting ask: grant exactly its act and target for some hours, "
                        "or decline it."
                    ),
                    scope=Scope.OWNER,
                ),
                Command(
                    name="grant",
                    module="mainboard.verbs.serve.asks",
                    help=(
                        "Grant one act within one scope for some hours, end such grants, or list "
                        "those in force."
                    ),
                    scope=Scope.OWNER,
                ),
                Command(
                    name="pending",
                    module="mainboard.verbs.serve.telemetry",
                    help="Show the spool lines no pass has delivered to the sink yet, by file.",
                    scope=Scope.READ,
                ),
                Command(
                    name="process",
                    module="mainboard.verbs.serve.telemetry",
                    help="Run one of serve's children in this process, as the supervisor does.",
                    show=False,
                ),
            ),
            children=(
                Node(
                    name="agents",
                    module="mainboard.verbs.serve.agents",
                    help="Point the coding agents' user-level telemetry at this workspace's sink.",
                    commands=(Command(name="sync", scope=Scope.OWNER),),
                ),
            ),
        ),
    ),
)
