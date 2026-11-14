"""Small isolated iOS benchmark project; no dependency on the SongMap app."""
import re


def project_text(bundle_id='com.danendra.SongFormerBenchmark', team_id=None):
    if not re.fullmatch(r'[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+', bundle_id):
        raise ValueError('Invalid bundle ID')
    if team_id and not re.fullmatch(r'[A-Z0-9]{10}', team_id):
        raise ValueError('Apple team ID must be 10 uppercase letters/digits')
    uid = lambda i: f'{i:024X}'
    obj = []
    def add(i, body): obj.append(f'{uid(i)} = {{{body}}};')
    assets = ['HeadFP32.mlmodelc', 'HeadINT8.mlmodelc', 'HeadPalette6.mlmodelc',
              'features.f32', 'reference_logits.f32', 'benchmark_manifest.json']
    sources = ['BenchmarkApp.swift', 'BenchmarkRunner.swift']
    for i, name in enumerate(sources):
        add(20+i, f'isa = PBXFileReference; lastKnownFileType = sourcecode.swift; path = "{name}"; sourceTree = "<group>";')
        add(40+i, f'isa = PBXBuildFile; fileRef = {uid(20+i)};')
    for i, name in enumerate(assets):
        file_type = 'folder' if name.endswith('mlmodelc') else 'file'
        add(60+i, f'isa = PBXFileReference; lastKnownFileType = {file_type}; path = "Assets/{name}"; sourceTree = "<group>";')
        add(80+i, f'isa = PBXBuildFile; fileRef = {uid(60+i)};')
    add(3, 'isa = PBXFileReference; explicitFileType = wrapper.application; path = Benchmark.app; sourceTree = BUILT_PRODUCTS_DIR;')
    add(4, f'isa = PBXGroup; children = ({uid(3)},); name = Products; sourceTree = "<group>";')
    children = ','.join(uid(20+i) for i in range(2))+','+','.join(uid(60+i) for i in range(6))+','+uid(4)
    add(5, f'isa = PBXGroup; children = ({children},); sourceTree = "<group>";')
    add(6, f'isa = PBXSourcesBuildPhase; buildActionMask = 2147483647; files = ({uid(40)},{uid(41)},); runOnlyForDeploymentPostprocessing = 0;')
    add(7, 'isa = PBXFrameworksBuildPhase; buildActionMask = 2147483647; files = (); runOnlyForDeploymentPostprocessing = 0;')
    add(8, 'isa = PBXResourcesBuildPhase; buildActionMask = 2147483647; files = ('+','.join(uid(80+i) for i in range(6))+',); runOnlyForDeploymentPostprocessing = 0;')
    add(2, f'isa = PBXNativeTarget; buildConfigurationList = {uid(10)}; buildPhases = ({uid(6)},{uid(7)},{uid(8)},); buildRules = (); dependencies = (); name = Benchmark; productName = Benchmark; productReference = {uid(3)}; productType = "com.apple.product-type.application";')
    add(1, f'isa = PBXProject; attributes = {{LastUpgradeCheck = 2700;}}; buildConfigurationList = {uid(9)}; compatibilityVersion = "Xcode 14.0"; developmentRegion = en; hasScannedForEncodings = 0; knownRegions = (en,Base,); mainGroup = {uid(5)}; productRefGroup = {uid(4)}; projectDirPath = ""; projectRoot = ""; targets = ({uid(2)},);')
    for i, name in [(11,'Debug'),(12,'Release')]:
        add(i, f'isa = XCBuildConfiguration; buildSettings = {{SDKROOT = iphoneos; IPHONEOS_DEPLOYMENT_TARGET = 18.0; CLANG_ENABLE_MODULES = YES;}}; name = {name};')
    signing = f'DEVELOPMENT_TEAM = {team_id};' if team_id else ''
    for i, name in [(13,'Debug'),(14,'Release')]:
        add(i, f'''isa = XCBuildConfiguration; buildSettings = {{
            PRODUCT_NAME = Benchmark; PRODUCT_BUNDLE_IDENTIFIER = {bundle_id};
            GENERATE_INFOPLIST_FILE = YES; INFOPLIST_KEY_UILaunchScreen_Generation = YES;
            INFOPLIST_KEY_UIApplicationSceneManifest_Generation = YES;
            CODE_SIGN_STYLE = Automatic; {signing} SWIFT_VERSION = 5.0;
            SWIFT_OPTIMIZATION_LEVEL = "-O"; SWIFT_STRICT_CONCURRENCY = targeted;
            TARGETED_DEVICE_FAMILY = "1,2"; SUPPORTED_PLATFORMS = "iphoneos iphonesimulator";
            IPHONEOS_DEPLOYMENT_TARGET = 18.0; CURRENT_PROJECT_VERSION = 1; MARKETING_VERSION = 1.0;
        }}; name = {name};''')
    for i, configs in [(9,(11,12)),(10,(13,14))]:
        add(i, f'isa = XCConfigurationList; buildConfigurations = ({uid(configs[0])},{uid(configs[1])},); defaultConfigurationIsVisible = 0; defaultConfigurationName = Release;')
    return '// !$*UTF8*$!\n{archiveVersion = 1; classes = {}; objectVersion = 56; objects = {\n'+'\n'.join(obj)+f'\n}}; rootObject = {uid(1)};}}\n'
