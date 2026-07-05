# scRNA Driver

This folder only contains BioPlexMDT driver code for calling an external scRNA/scMulan workflow.

Place scMulan source code and model weights outside this repository, then set:

```text
BIOPLEX_SCRNA_EXTERNAL_PIPELINE=
BIOPLEX_SCRNA_TOOLS_DIR=tools/scrna
```

Default BioPlexMDT entry:

```text
tools/scrna/integrated_analysis_pipeline.py
```

The previous local classifier weight and scMulan checkpoint were moved out of the GitHub-ready project with the other tool contents.




