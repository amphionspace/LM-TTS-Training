import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts.monitor_training import main


class MonitorHandoffTests(unittest.TestCase):
    def prepare(self, folder):
        old, new = Path(folder) / 'token', Path(folder) / 'sqrt'
        for run, pid in ((old, 10), (new, 20)):
            run.mkdir()
            (run / 'supervision-prompt.md').write_text('Review {{SNAPSHOT}}')
            (run / 'training-process.json').write_text(json.dumps({'pid': pid, 'start_ticks': str(pid)}))
            (run / 'final-verification.json').write_text(json.dumps({'passed': True, 'pid': pid}))
        (old / 'supervision').mkdir()
        (old / 'supervision/session-id').write_text('existing-session\n')
        (old / 'next-run.json').write_text(json.dumps({'run_dir': str(new)}))
        return old, new

    def run_monitor(self, old, once):
        calls = []

        def review(command, **kwargs):
            calls.append(command)
            kwargs['stdout'].write(json.dumps({'type': 'thread.started', 'thread_id': 'existing-session'}) + '\n')
            return MagicMock(returncode=0)

        def snapshot(run, previous):
            pid = json.loads((run / 'training-process.json').read_text())['pid']
            return {'time': '2026-09-10T00:00:00+00:00', 'last_step': 19000,
                    'process': {'pid': pid}, 'training_exit': {'pid': pid, 'exit_code': 0}}

        argv = ['monitor', '--run-dir', str(old)] + (['--once'] if once else [])
        with patch.object(sys, 'argv', argv), patch('scripts.monitor_training.collect_snapshot', side_effect=snapshot), \
             patch('scripts.monitor_training.subprocess.Popen', side_effect=review):
            main()
        return calls

    def test_completion_waits_for_verified_matching_successor(self):
        for ack in (None, {'status': 'verified', 'pid': 20, 'start_ticks': 'wrong'}):
            with self.subTest(ack=ack), tempfile.TemporaryDirectory() as folder:
                old, new = self.prepare(folder)
                if ack:
                    (old / 'next-run-ack.json').write_text(json.dumps({**ack, 'run_dir': str(new)}))
                self.assertEqual(len(self.run_monitor(old, once=True)), 1)
                status = json.loads((old / 'supervision/status.json').read_text())
                self.assertEqual(status['status'], 'handoff_pending')
                self.assertFalse((new / 'supervision').exists())

    def test_verified_handoff_continues_same_session_on_successor(self):
        with tempfile.TemporaryDirectory() as folder:
            old, new = self.prepare(folder)
            (old / 'next-run-ack.json').write_text(json.dumps({
                'status': 'verified', 'pid': 20, 'start_ticks': '20', 'run_dir': str(new)}))
            calls = self.run_monitor(old, once=False)
            self.assertEqual(len(calls), 2)
            for command in calls:
                self.assertEqual(command[-3:], ['resume', 'existing-session', '-'])
            self.assertEqual((new / 'supervision/session-id').read_text().strip(), 'existing-session')
            self.assertEqual(json.loads((old / 'supervision/status.json').read_text())['status'], 'completed')
            self.assertEqual(json.loads((new / 'supervision/status.json').read_text())['status'], 'completed')

    def test_successor_ack_cannot_skip_predecessor_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            old, new = self.prepare(folder)
            (old / 'final-verification.json').write_text(json.dumps({'passed': False, 'pid': 10}))
            (old / 'next-run-ack.json').write_text(json.dumps({
                'status': 'verified', 'pid': 20, 'start_ticks': '20', 'run_dir': str(new)}))
            self.assertEqual(len(self.run_monitor(old, once=True)), 1)
            self.assertEqual(json.loads((old / 'supervision/status.json').read_text())['status'], 'monitoring')
            self.assertFalse((new / 'supervision').exists())
