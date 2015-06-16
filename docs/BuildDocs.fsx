#I "../packages/FSharp.Compiler.Service.0.0.87/lib/net40"
#r "FSharp.Compiler.Service.dll"
#I "../packages/FSharpVSPowerTools.Core.1.8.0/lib/net45"
#r "FSharpVSPowerTools.Core.dll"
#I "../packages/FSharp.Formatting.2.9.8/lib/net40"
#r "CSharpFormat.dll"
#r "FSharp.CodeFormat.dll"
#r "FSharp.Literate.dll"
#r "FSharp.MetadataFormat.dll"
#r "FSharp.Markdown.dll"

open System.IO
open FSharp.Literate
open FSharp.MetadataFormat

//
// Setup output directory structure and copy static files
//

let source = __SOURCE_DIRECTORY__ 
let docs = Path.Combine(source, "")
let relative subdir = Path.Combine(docs, subdir)

if not (Directory.Exists(relative "output")) then
    Directory.CreateDirectory(relative "output") |> ignore
if not (Directory.Exists(relative "output/img")) then
    Directory.CreateDirectory (relative "output/img") |> ignore
if not (Directory.Exists(relative "output/misc")) then
    Directory.CreateDirectory (relative "output/misc") |> ignore
if not (Directory.Exists(relative "output/reference")) then
    Directory.CreateDirectory (relative "output/reference") |> ignore

for fileInfo in DirectoryInfo(relative "input/files/misc").EnumerateFiles() do
    fileInfo.CopyTo(Path.Combine(relative "output/misc", fileInfo.Name), true) |> ignore

for fileInfo in DirectoryInfo(relative "input/files/img").EnumerateFiles() do
    fileInfo.CopyTo(Path.Combine(relative "output/img", fileInfo.Name), true) |> ignore

//
// Generate documentation
//

let tags = ["project-name", "Hype"; "project-author", "Atılım Güneş Baydin"; "project-github", "http://github.com/gbaydin/Hype"; "project-nuget", "https://www.nuget.org/packages/hype"; "root", ""]

Literate.ProcessScriptFile(relative "input/index.fsx", relative "input/templates/template.html", relative "output/index.html", replacements = tags)
Literate.ProcessScriptFile(relative "input/download.fsx", relative "input/templates/template.html", relative "output/download.html", replacements = tags)

//
// Generate API reference
//

let library = relative "../src/Hype/bin/Debug/Hype.dll"
let layoutRoots = [relative "input/templates"; relative "input/templates/reference" ]

MetadataFormat.Generate(library, relative "output/reference", layoutRoots, tags, markDownComments = true, libDirs = [relative "../src/Hype/bin/Debug/"])
