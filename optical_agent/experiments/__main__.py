"""Run one explicitly selected experiment; no eager optional imports."""
import argparse
import json
import runpy
import sys

COMMANDS = {'build_rag_questions': 'optical_agent.experiments.build_rag_questions', 'calibrate_detector': 'optical_agent.experiments.calibrate_detector', 'calibrate_quality': 'optical_agent.experiments.calibrate_quality', 'demo_agent': 'optical_agent.experiments.demo_agent', 'demo_independent_eval': 'optical_agent.experiments.demo_independent_eval', 'demo_langgraph': 'optical_agent.experiments.demo_langgraph', 'demo_mcp': 'optical_agent.experiments.demo_mcp', 'demo_memory': 'optical_agent.experiments.demo_memory', 'demo_task_validation': 'optical_agent.experiments.demo_task_validation', 'diagnose_robot_training': 'optical_agent.experiments.diagnose_robot_training', 'download_underwater_data': 'optical_agent.experiments.download_underwater_data', 'download_vision_data': 'optical_agent.experiments.download_vision_data', 'evaluate_adaptive': 'optical_agent.experiments.evaluate_adaptive', 'evaluate_agent': 'optical_agent.experiments.evaluate_agent', 'evaluate_detector': 'optical_agent.experiments.evaluate_detector', 'evaluate_orchestration': 'optical_agent.experiments.evaluate_orchestration', 'evaluate_quality_pairs': 'optical_agent.experiments.evaluate_quality_pairs', 'evaluate_rag': 'optical_agent.experiments.evaluate_rag', 'evaluate_underwater_illumination': 'optical_agent.experiments.evaluate_underwater_illumination', 'evaluate_vision_detectors': 'optical_agent.experiments.evaluate_vision_detectors', 'evaluate_vision_exposure': 'optical_agent.experiments.evaluate_vision_exposure', 'prepare_rag': 'optical_agent.experiments.prepare_rag', 'prepare_rag_corpus': 'optical_agent.experiments.prepare_rag_corpus', 'prepare_vision_data': 'optical_agent.experiments.prepare_vision_data', 'publish_rag': 'optical_agent.experiments.publish_rag', 'reconcile_vision_calibration': 'optical_agent.experiments.reconcile_vision_calibration', 'report_adaptive': 'optical_agent.experiments.report_adaptive', 'report_vision': 'optical_agent.experiments.report_vision', 'summarize_formal_eval': 'optical_agent.experiments.summarize_formal_eval', 'train_detector': 'optical_agent.experiments.train_detector', 'train_vision_detectors': 'optical_agent.experiments.train_vision_detectors', 'verify_vision_web': 'optical_agent.experiments.verify_vision_web'}

NO_ARGUMENT_COMMANDS = ['build_rag_questions', 'diagnose_robot_training', 'evaluate_vision_detectors', 'prepare_rag_corpus', 'publish_rag', 'reconcile_vision_calibration', 'report_vision', 'verify_vision_web']

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, epilog='Pass --help after a command for its original options.')
    parser.add_argument('--list', action='store_true', help='Print the fixed command registry as JSON')
    parser.add_argument('command', nargs='?', choices=sorted(COMMANDS))
    if not argv or argv[0] in {'--help', '-h'}:
        parser.print_help()
        return
    if argv[0] == '--list':
        args = parser.parse_args(argv)
        if args.list:
            print(json.dumps(COMMANDS, sort_keys=True))
        else:
            parser.print_help()
        return
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        parser.error('Unknown experiment: ' + command)
    if command in NO_ARGUMENT_COMMANDS and rest:
        if rest == ['--help'] or rest == ['-h']:
            print(command + ': original no-argument script. Run explicitly without flags. Historical data/version prerequisites apply.')
            return
        parser.error(command + ' accepts no arguments; use --help for prerequisites')
    previous = sys.argv
    try:
        sys.argv = [command] + rest
        runpy.run_module(COMMANDS[command], run_name='__main__')
    finally:
        sys.argv = previous

if __name__ == '__main__':
    main()
