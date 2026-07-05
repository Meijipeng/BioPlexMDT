# RaSPr Driver

This folder only contains BioPlexMDT driver code for calling an external RaSPr / DICOM workflow.

Place the real RaSPr workflow outside this repository, then set:

```text
BIOPLEX_RASPR_EXTERNAL_SCRIPT=
BIOPLEX_RASPR_SCRIPT_PATH=tools/raspr/run_case_all.sh
```

Default BioPlexMDT entry:

```text
tools/raspr/run_case_all.sh
```




