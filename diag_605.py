"""Diagnostic script: run full pipeline from 140 to see 605 with post-skip reset + wider AOI."""
import sys
print("diag_605.py starting...", flush=True)

import project_sgil.constants as C
C.PLOT = False
C.PLOT_WEDGES = False

from project_sgil.automated_sgil import AutomatedSGIL
print("imports done", flush=True)

sgil = AutomatedSGIL(start_index=140, specific_indices=None)
print("AutomatedSGIL created, running from 140...", flush=True)
results = sgil.run()

# Print just frame 605's result
for r in results:
    if r.image_name and "605" in r.image_name:
        est = r.estimated_pose
        est_str = f"({est.x:.2f}, {est.y:.2f})" if est else "None"
        print(f"\n[605 RESULT] matched={r.matched} "
              f"sgil_err={r.sgil_err_m:.2f}m "
              f"est_pose={est_str}",
              flush=True)
        break

print("done", flush=True)
