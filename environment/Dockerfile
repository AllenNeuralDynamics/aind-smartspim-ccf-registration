FROM registry.codeocean.allenneuraldynamics.org/published/1d42f62b-68e4-40b7-ab32-32225960c6a3:v1

# Build on the published image of capsule 1d42f62b (this capsule's upstream) instead
# of reinstalling its dependencies.
#
# The previous base was codeocean/jupyterlab:...python3.8..., followed by a pip
# install of a 2023-era pin list. That list no longer builds. cloud-volume's
# all_codecs extra pulls zfpy, which has no wheel for this platform, so pip compiles
# it under a build-isolated Cython 3, and the old Cython 2 syntax fails:
#
#     python/zfpy.pyx:93:28: 'ndarray' is not a type identifier
#
# Nothing about that is specific to this capsule's code -- a fresh build of upstream
# main fails identically. The pins were correct when written and rotted underneath.
#
# The pipeline never hit this because it does not rebuild: its registration node
# pulls exactly this published image. Code Ocean images carry the environment only
# -- the pipeline clones the capsule repo into capsule/code at run time -- so using
# it as a base gives the known-good dependency set while this capsule's own code
# still runs on top.
#
# Two consequences worth knowing. There is no pip install here, so nothing is
# fetched over git during the build, which is why the ARG GIT_ASKPASS /
# COPY git-ask-pass lines are gone: they existed to authenticate the
# aicsimageio@git+... dependency. And the conda env this image ships, ccf_reg, is
# the one code/run activates, so that contract is unchanged.
#
# The aicsimageio pin is worth revisiting separately. It points at a branch of a
# personal fork of an org fork of an archived repository, to supply an OmeZarrWriter
# that ome-zarr -- already in this environment -- now provides itself.

ARG DEBIAN_FRONTEND=noninteractive
