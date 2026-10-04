"""Delivery guards added after the frozen single-frame experiment.

The evaluated OpticalAgent implementation remains unchanged and archived.
"""
import copy

from agent import OpticalAgent, _consistent, _lost_boxes, _overlay

DELIVERY_REVISION = 'v040_delivery_r1'


class DeliveryOpticalAgent(OpticalAgent):
    def inspect(self, images):
        if len(images) <= 1:
            return super().inspect(images)
        if any(image.shape != images[0].shape for image in images):
            raise ValueError('Multi-exposure images must have identical dimensions and be aligned')
        runs = [self._one(image) for image in images]
        valid = [(i, r) for i, (r, _) in enumerate(runs) if r['quality_selected']['quality_pass'] is True]
        pool = valid or [(i, r) for i, (r, _) in enumerate(runs)]
        selected, report = min(pool, key=lambda item: (item[1]['quality_selected']['lost_tile_fraction'],
                                                      -item[1]['quality_selected']['global']['contrast']))
        output = copy.deepcopy(report)
        stable = bool(valid) and all(r['status'] == 'reliable' for _, r in valid)
        if stable:
            stable = all(_consistent(valid[0][1]['accepted_detections'], r['accepted_detections'])
                         for _, r in valid[1:])
        if not stable:
            output['status'] = 'unreliable' if valid else 'quality_failure'
            output['target_conclusion'] = 'unreliable'
            if output['defect_conclusion'] is not None:
                output['defect_conclusion'] = 'unreliable'
            output['next_action'] = 'provide_more_aligned_exposure_frames'
            output['correction']['accepted'] = False
            output['accepted_detections'] = []
            output['tentative_detections'] = copy.deepcopy(output['detections_selected'])
            output['reasons'].append('质量有效帧中存在不可靠或冲突的检测证据，不能只取其中可靠帧宣称一致')
        output['multi_frame'] = {
            'frame_count': len(images), 'selected_frame': selected,
            'frame_statuses': [r['status'] for r, _ in runs],
            'frame_summaries': [{'index': i, 'status': r['status'],
                'exposure_state': r['quality_before']['exposure_state'],
                'lost_tile_fraction_before': r['quality_before']['lost_tile_fraction'],
                'lost_tile_fraction_selected': r['quality_selected']['lost_tile_fraction'],
                'accepted_detection_count': len(r['accepted_detections'])} for i, (r, _) in enumerate(runs)],
            'consistent_frames': len(valid) if stable else 0,
            'mode': 'all_quality_valid_frames_cross_check' if len(valid) > 1 else 'single_valid_frame_selection',
            'evidence_scope': 'cross_frame_rule_checks' if stable and len(valid) > 1 else 'no_verified_cross_exposure_consensus',
            'alignment': 'assumed_by_input_contract_not_geometrically_verified',
            'runtime_revision': DELIVERY_REVISION,
        }
        output['latency_ms'] = round(sum(r['latency_ms'] for r, _ in runs), 1)
        views = dict(runs[selected][1])
        if not stable:
            # Final accepted-box view must match the emptied accepted list.
            # Original/corrected candidate views remain available for inspection.
            views['detections_selected'] = _overlay(views['selected'], [], _lost_boxes(output['quality_selected']))
        return output, views
