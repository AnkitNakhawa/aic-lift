# MuJoCo Mesh Export Pipeline

Converts the AIC Gazebo world into MuJoCo-compatible mesh assets (`.obj`, `.stl`, `.png`) for use in `aic_utils/aic_mujoco/mjcf/`.

The mesh files are **gitignored** — this pipeline is how you (re)generate them whenever the Gazebo world or asset models change.

---

## Overview

The pipeline does the following inside the `aic_eval` Docker container:

1. Starts Gazebo headless with the full task-board scene (task board, NIC card mount, SFP/SC cable, robot)
2. Waits for Gazebo to fully load and write its world SDF to `/tmp/aic.sdf`
3. Installs `sdformat_mjcf` (the `aic` branch of `gz-mujoco`) with compatibility patches for the container's library versions
4. Fixes known URI issues in the SDF
5. Runs `sdf2mjcf` which converts each `.glb` mesh to one or more `.obj` + `.png` files via `trimesh`
6. Outputs ~73 mesh/texture files ready to drop into `aic_utils/aic_mujoco/mjcf/`

---

## Option A — GitHub Actions (recommended for fresh machines)

**Pre-requisites:** write access to the repo (to trigger `workflow_dispatch`).

1. Go to **Actions → Export MuJoCo Mesh Assets → Run workflow**
2. Wait ~15–20 min for the run to complete
3. Download the `mujoco-mesh-assets` artifact from the completed run
4. Unzip into your local `aic_utils/aic_mujoco/mjcf/`:

```bash
unzip mujoco-mesh-assets.zip -d aic_utils/aic_mujoco/mjcf/
```

> The artifact zip excludes `.xml` files — it only contains mesh/texture assets so it won't clobber your existing `scene.xml`, `aic_robot.xml`, or `aic_world.xml`.

---

## Option B — Run locally on a VM that already has `aic_eval` running

Use this when you already have the `aic_eval` container up (e.g. a GCP VM from `./scripts/start_eval.sh`). The container has Gazebo running so the 120 s wait is already done.

### Pre-requisites

- `aic_eval` container is running (`sudo docker ps | grep aic_eval`)
- `sudo` docker access on the VM
- Internet access to clone `gz-mujoco`

### Steps

**1. Clone the `aic` branch of `gz-mujoco` and copy it into the container:**

```bash
git clone --depth 1 -b aic https://github.com/gazebosim/gz-mujoco.git /tmp/gz-mujoco
sudo docker cp /tmp/gz-mujoco/sdformat_mjcf aic_eval:/tmp/sdformat_mjcf
```

**2. Install sdformat deps and `sdformat_mjcf` inside the container:**

```bash
sudo docker exec aic_eval bash -c "apt-get update -q && apt-get install -y libsdformat16 python3-sdformat16 python3-gz-math9"

sudo docker exec aic_eval bash -c "
  echo '__version__ = \"0.0.1\"' > /tmp/sdformat_mjcf/src/sdformat_mjcf/__version__.py
  cd /tmp/sdformat_mjcf && python3 setup.py install 2>&1 | tail -5
"
```

**3. Overwrite the installed egg with the full aic branch source** (brings in `mesh_io.py` for `.glb → .obj` conversion):

```bash
sudo docker exec aic_eval bash -c "
  SRC=/tmp/sdformat_mjcf/src/sdformat_mjcf
  DST=\$(python3 -c 'import sdformat_mjcf; import os; print(os.path.dirname(sdformat_mjcf.__file__))')
  cp -r \$SRC/sdformat_to_mjcf/mesh_io.py         \$DST/sdformat_to_mjcf/
  cp -r \$SRC/sdformat_to_mjcf/converters/         \$DST/sdformat_to_mjcf/
  cp -r \$SRC/sdformat_to_mjcf/sdformat_to_mjcf.py  \$DST/sdformat_to_mjcf/
  cp -r \$SRC/sdformat_to_mjcf/sdf_kinematics.py    \$DST/sdformat_to_mjcf/
  cp -r \$SRC/utils/                                \$DST/
"
```

**4. Patch version-name mismatches** (the aic branch was written against `sdformat13` / `gz.math7`; the container ships them as `sdformat` / `gz.math`):

```bash
sudo docker exec aic_eval bash -c "
  DST=\$(python3 -c 'import sdformat_mjcf; import os; print(os.path.dirname(sdformat_mjcf.__file__))')
  find \$DST -name '*.py' | xargs sed -i \
    's/from sdformat13 /from sdformat /g
     s/import sdformat13/import sdformat/g
     s/gz\.math7/gz.math/g
     s/gz\.math8/gz.math/g
     s/gz\.math9/gz.math/g'
  find \$DST -name '*.pyc' -delete
"
```

**5. Export the SDF from the running Gazebo world:**

```bash
# The WorldSdfGeneratorPlugin writes it to /tmp/aic.sdf automatically on startup.
# Verify it's there:
sudo docker exec aic_eval ls -lh /tmp/aic.sdf
```

If it's missing, trigger it manually:
```bash
sudo docker exec aic_eval bash -c "
  source /opt/ros/kilted/setup.bash && source /ws_aic/install/setup.bash
  gz service -s /world/aic_world/generate_world_sdf --reqtype gz.msgs.SdfGeneratorConfig \
    --reptype gz.msgs.StringMsg --timeout 10000 --req '{}' > /tmp/gz_sdf_out.txt
  python3 /tmp/parse_sdf.py < /tmp/gz_sdf_out.txt
"
```

**6. Fix SDF URI issues:**

```bash
sudo docker exec aic_eval bash -c "
  sed -i 's|file://<urdf-string>/model://|file:///|g' /tmp/aic.sdf
  ASSETS=/ws_aic/install/share/aic_assets/models
  sed -i \"s|file:///sc_plug_visual|file://\${ASSETS}/SC Plug/sc_plug_visual|g\" /tmp/aic.sdf
  sed -i \"s|file:///lc_plug_visual|file://\${ASSETS}/LC Plug/lc_plug_visual|g\" /tmp/aic.sdf
  sed -i \"s|file:///sfp_module_visual|file://\${ASSETS}/SFP Module/sfp_module_visual|g\" /tmp/aic.sdf
  sed -i 's|<uri>file://\([^<]*\)</uri>|<uri>\1</uri>|g' /tmp/aic.sdf
"
```

**7. Convert SDF → MJCF + mesh assets:**

```bash
sudo docker exec aic_eval bash -c "
  mkdir -p /tmp/aic_mujoco_world
  source /opt/ros/kilted/setup.bash && source /ws_aic/install/setup.bash
  PATH=/usr/local/bin:\$PATH sdf2mjcf /tmp/aic.sdf /tmp/aic_mujoco_world/aic_world.xml
"
# Should print ~73 'Exported ...' lines and exit 0
```

**8. Copy assets to the repo:**

```bash
sudo docker cp aic_eval:/tmp/aic_mujoco_world /tmp/aic_mujoco_world
rsync -av --exclude='*.xml' /tmp/aic_mujoco_world/ aic_utils/aic_mujoco/mjcf/
```

---

## Compatibility notes

The pipeline works around three mismatches between the `aic` branch of `gz-mujoco` and the libraries shipped in the `aic_eval` container:

| What the code expects | What the container ships | Fix applied |
|---|---|---|
| `import sdformat13` | module is `sdformat` (from `python3-sdformat16`) | `sed` all `.py` files in the installed package |
| `from gz.math7 import` | module is `gz.math` (no version suffix) | same `sed` pass |
| `setup.py` installs old `geometry.py` (no `.glb` support) | aic branch has `mesh_io.py` + updated `geometry.py` | overwrite installed files from source after `setup.py install` |
| `<uri>file:///abs/path</uri>` | `os.path.join` treats it as a relative path → broken | strip `file://` to leave plain absolute paths |

If the `aic_eval` image is ever updated and these mismatches are resolved upstream, the patching steps can be removed.

---

## Output

All files land in `aic_utils/aic_mujoco/mjcf/` alongside the existing XML files:

| Type | Count | Source |
|---|---|---|
| `.obj` | ~61 | `.glb` meshes converted via `trimesh` |
| `.stl` | ~7 | UR5e arm links (already STL in Gazebo) |
| `.png` | ~7 | Textures extracted from `.glb` materials |

These files are gitignored (`aic_utils/aic_mujoco/mjcf/*.obj` etc.) — regenerate them with this pipeline whenever the Gazebo world changes.
