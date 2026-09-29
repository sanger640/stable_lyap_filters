# FIGNet* port (Program B stage 1)

Upstream: https://github.com/jongyaoY/fignet (MIT), pinned at
`05999631006e472b875751f585ef3d5c6e5602e3`. The author (Zongyao Yi) also wrote Act-FIGNet.

Setup:

```bash
git clone https://github.com/jongyaoY/fignet external/fignet
cd external/fignet && git checkout 05999631006e472b875751f585ef3d5c6e5602e3
git apply ../../vendor/fignet_port.patch
pip install -e . --no-deps
```

`fignet_port.patch` only removes heavy or unavailable dependencies. Model code is unchanged.

- hpp-fcl -> coal 3.0.3 (renames `Vec3f`/`Transform3f` to `Vec3s`/`Transform3s`); hpp-fcl 2.4.4
  would downgrade numpy.
- torch_scatter `scatter(reduce="sum", dim=0)` -> `index_add_` (`fignet/_compat.py`).
- pytorch3d `corresponding_points_alignment` -> batched Kabsch (`fignet/_compat.py`).
- robosuite is imported lazily, since it is needed for rendering only.
