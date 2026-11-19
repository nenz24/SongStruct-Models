import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import venv


class ContextNotebookEnvironmentTests(unittest.TestCase):
    def test_interpreter_keeps_virtual_environment_when_python_is_symlink(self):
        notebook = Path(__file__).resolve().parents[2] / 'section_context_app_install.ipynb'
        cells = json.loads(notebook.read_text())['cells']
        setup = next(''.join(c['source']) for c in cells
                     if c['cell_type'] == 'code' and 'RUN_PYTHON=' in ''.join(c['source']))
        assignment = next(node for node in ast.parse(setup).body
                          if isinstance(node, ast.Assign) and any(
                              isinstance(t, ast.Name) and t.id == 'RUN_PYTHON' for t in node.targets))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'SongMap'
            root.mkdir()
            environment = root.parent / 'Section-Annotations/.venv-songformer311'
            venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
            scope = {'ROOT': root}
            exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(notebook), 'exec'), scope)
            python = scope['RUN_PYTHON']
            self.assertTrue(python.is_symlink())
            self.assertEqual(python.parent.parent, environment)
            actual = json.loads(subprocess.check_output(
                [str(python), '-c', 'import sys,json; print(json.dumps([sys.executable,sys.prefix,sys.base_prefix]))'],
                text=True))
            self.assertEqual(Path(actual[0]), python)
            self.assertEqual(Path(actual[1]).resolve(), environment.resolve())
            self.assertNotEqual(actual[1], actual[2])


if __name__ == '__main__':
    unittest.main()
