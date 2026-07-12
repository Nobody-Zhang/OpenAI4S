"use strict";

const assert = require("node:assert/strict");
const renderer = require("../openai4s/server/webui/scientific_renderers.js");

const fasta = renderer.parseSequence(">alpha first\nACGTACGT\n>beta\nACGU\n", "reads.fasta");
assert.equal(fasta.format, "FASTA");
assert.equal(fasta.records.length, 2);
assert.equal(fasta.total_length, 12);
assert.equal(fasta.records[0].description, "first");

const alignment = renderer.parseAlignment(
  "CLUSTAL W\n\nseq1    AC-GT\nseq2    ACTGT\n        ** **\n\nseq1    AA\nseq2    A-\n",
  "example.aln",
);
assert.equal(alignment.format, "Clustal");
assert.deepEqual(alignment.records.map((record) => record.sequence), ["AC-GTAA", "ACTGTA-"]);
assert.equal(alignment.columns, 7);

const genome = renderer.parseGenome(
  "chr1\t10\t25\tfeature-a\nchr1\t30\t45\tfeature-b\nchr2\t5\t9\tfeature-c\n",
  "track.bed",
);
assert.equal(genome.format, "BED");
assert.equal(genome.features.length, 3);
assert.equal(genome.chromosomes.length, 2);

const molfile = [
  "Water",
  "  OpenAI4S",
  "",
  "  3  2  0  0  0  0            999 V2000",
  "    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0",
  "   -0.8000   -0.6000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0",
  "    0.8000   -0.6000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0",
  "  1  2  1  0  0  0  0",
  "  1  3  1  0  0  0  0",
  "M  END",
].join("\n");
const molecule = renderer.parseMolfile(molfile);
assert.equal(molecule.title, "Water");
assert.equal(molecule.atoms.length, 3);
assert.equal(molecule.bonds.length, 2);

const latex = renderer.latexPreview("\\section{Result}\nThe value is $$\\alpha \\leq 1$$.");
assert.deepEqual(latex[0], { kind: "heading", level: 1, text: "Result" });
assert.equal(latex.some((block) => block.kind === "math" && block.text.includes("α")), true);

const catalog = [{ renderer_id: "sequence" }, { renderer_id: "download" }];
assert.equal(renderer.rendererIdFromDescriptor({ renderer: { renderer_id: "sequence" } }, catalog), "sequence");
assert.equal(renderer.rendererIdFromDescriptor({ renderer: { renderer_id: "unknown-script" } }, catalog), "download");

console.log("scientific renderer parser smoke: ok");
