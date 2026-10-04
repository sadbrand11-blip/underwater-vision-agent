"""Protocol test process only; not the released real-detector entry point."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optical_agent.adaptive_goals import CLASSES
from optical_agent.mcp_adapter import VisionMCPAdapter
from optical_agent.mcp_server import configure_logging, create_server


class FixtureDetector:
    dataset, threshold, calibration = 'sodd', .35, None
    classes = ('background', *CLASSES)

    def predict(self, image):
        print('FIXTURE_PRINT_MUST_STAY_OFF_PROTOCOL')
        return [{'class_name': 'pipe', 'class_id': 6, 'score': .8, 'box': [4, 4, 16, 16]},
                {'class_name': 'qr_codes', 'class_id': 5, 'score': .7, 'box': [20, 20, 32, 32]}]


if __name__ == '__main__':
    root = Path(sys.argv[1]).resolve()
    configure_logging(root / 'protocol.log')
    create_server(VisionMCPAdapter(FixtureDetector, data_root=root, output_root=root / 'outputs')).run(transport='stdio')
