"""Decode, validate, globally group and freeze the local experiment split."""
import argparse
from optical_agent.vision_data import ROOT, prepare, load_manifest, audit_source_annotations, quarantine_invalid_annotations
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--audit-source-only',action='store_true')
    p.add_argument('--quarantine-invalid',action='store_true')
    a=p.parse_args()
    if a.quarantine_invalid:
        quarantine_invalid_annotations()
        raise SystemExit(0)
    audit_source_annotations()
    if not a.audit_source_only:
        if (ROOT/'vision_v040/manifest.json').exists():
            load_manifest(ROOT/'vision_v040/manifest.json')
            print('Reusing the frozen experiment split; new experiments require a new version directory',flush=True)
        else:
            prepare()
        quarantine_invalid_annotations()
