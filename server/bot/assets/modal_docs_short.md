# Modal llms-full.txt

> Modal is a platform for running Python code in the cloud with minimal
> configuration, especially for serving AI models and high-performance batch
> processing. It supports fast prototyping, serverless APIs, scheduled jobs,
> GPU inference, distributed volumes, and sandboxes.

Important notes:

- Modal's primitives are embedded in Python and tailored for AI/GPU use cases,
    but they can be used for general-purpose cloud compute.
- Modal is a serverless platform, meaning you are only billed for resources used
    and can spin up containers on demand in seconds.

You can sign up for free at [https://modal.com] and get $30/month of credits.

## Guides

### Custom container images

#### Defining Images

# Images

This guide walks you through how to define a Modal Image, the environment your Modal code runs in.

The typical flow for defining an Image in Modal is
[method chaining](https://jugad2.blogspot.com/2016/02/examples-of-method-chaining-in-python.html)
starting from a base Image, like this:

```python
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install("torch<3")
    .env({"HALT_AND_CATCH_FIRE": "0"})
    .run_commands("git clone https://github.com/modal-labs/agi && echo 'ready to go!'")
)
```

If you have your own container image defintions, like a Dockerfile or a registry link, you can use those too!
See [this guide](https://modal.com/docs/guide/existing-images).

This page is a high-level guide to using Modal Images.
For reference documentation on the `modal.Image` object, see
[this page](https://modal.com/docs/reference/modal.Image).

## What are Images?

Your code on Modal runs in _containers_. Containers are like light-weight
virtual machines -- container engines use
[operating system tricks](https://earthly.dev/blog/chroot/) to isolate programs
from each other ("containing" them), making them work as though they were
running on their own hardware with their own filesystem. This makes execution
environments more reproducible, for example by preventing accidental
cross-contamination of environments on the same machine. For added security,
Modal runs containers using the sandboxed
[gVisor container runtime](https://cloud.google.com/blog/products/identity-security/open-sourcing-gvisor-a-sandboxed-container-runtime).

Containers are started up from a stored "snapshot" of their filesystem state
called an _image_. Producing the image for a container is called _building_ the
image.

By default, Modal Functions and Sandboxes run in a
[Debian Linux](https://en.wikipedia.org/wiki/Debian) container with a basic
Python installation of the same minor version `v3.x` as your local Python
interpreter.

To make your Apps and Functions useful, you will probably need some third party system packages
or Python libraries. Modal provides a number of options to customize your container images at
different levels of abstraction and granularity, from high-level convenience
methods like `pip_install` through wrappers of core container image build
features like `RUN` and `ENV`. We'll cover each of these in this guide,
along with tips and tricks for building Images effectively when using each tool.

## Add Python packages

The simplest and most common Image modification is to add a third party
Python package, like [`pandas`](https://pandas.pydata.org/).

You can add Python packages to the environment by passing all the packages you
need to the [`Image.uv_pip_install`](https://modal.com/docs/reference/modal.Image#uv_pip_install) method,
which installs packages with [`uv`](https://docs.astral.sh/uv/):

```python
import modal

datascience_image = (
    modal.Image.debian_slim()
    .uv_pip_install("pandas==2.2.0", "numpy")
)

@app.function(image=datascience_image)
def my_function():
    import pandas as pd
    import numpy as np

    df = pd.DataFrame()
    ...
```

You can include
[Python dependency version specifiers](https://peps.python.org/pep-0508/),
like `"torch<3"`, in the arguments. But we recommend pinning dependencies
tightly, like `"torch==2.8.0"`, to improve the reproducibility and robustness
of your builds.

If you run into any issues with
[`Image.uv_pip_install`](https://modal.com/docs/reference/modal.Image#uv_pip_install), then
you can fallback to [`Image.pip_install`](https://modal.com/docs/reference/modal.Image#pip_install) which
uses standard [`pip`](https://pip.pypa.io/en/stable/user_guide/):

```python
datascience_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("pandas==2.2.0", "numpy")
)
```

Note that because you can define a different environment for each and every
function if you so choose, you don't need to worry about virtual
environment management. Containers make for much better separation of concerns!

If you want to run a specific version of Python remotely rather than just
matching the one you're running locally, provide the `python_version` as a
string when constructing the base image, like we did above.

## Add local files with `add_local_dir` and `add_local_file`

Sometimes your containers need a dependency that's not available on the Internet,
like configuration files or code on your laptop.

To forward files from your local system use the
`image.add_local_dir` and `image.add_local_file` Image methods.

```python
image = modal.Image.debian_slim().add_local_dir("/user/erikbern/.aws", remote_path="/root/.aws")
```

By default, these files are added to your container as it starts up rather than introducing
a new Image layer. This means that the redeployment after making changes is really quick, but
also means you can't run additional build steps after. You can specify a `copy=True` argument
to the `add_local_` methods to instead force the files to be included in the built Image.

### Add local Python code with `add_local_python_source`

You can add Python code that's importable locally to your container
by providing the module name to
[`Image.add_local_python_source`](https://modal.com/docs/reference/modal.Image#add_local_python_source).

```python
image_with_module = modal.Image.debian_slim().add_local_python_source("local_module")

@app.function(image=image_with_module)
def f():
    import local_module

    local_module.do_stuff()
```

The difference from `add_local_dir` is that `add_local_python_source` takes module names as arguments
instead of a file system path and looks up the local package's or module's location via Python's importing
mechanism. The files are then added to directories that make them importable in containers in the
same way as they are locally.

This is intended for pure Python auxiliary modules that are part of your project and that your code imports.
Third party packages should be installed via
[`Image.uv_pip_install`](https://modal.com/docs/reference/modal.Image#uv_pip_install) or similar.

### What if I have different Python packages locally and remotely?

You might want to use packages inside your Modal code that you don't have on
your local computer. In the example above, we build a container that uses
`pandas`. But if we don't have `pandas` locally, on the computer building the
Modal App, we can't put `import pandas` at the top of the script, since it would
cause an `ImportError`.

The easiest solution to this is to put `import pandas` in the function body
instead, as you can see above. This means that `pandas` is only imported when
running inside the remote Modal container, which has `pandas` installed.

Be careful about what you return from Modal Functions that have different
packages installed than the ones you have locally! Modal Functions return Python
objects, like `pandas.DataFrame`s, and if your local machine doesn't have
`pandas` installed, it won't be able to handle a `pandas` object (the error
message you see will mention
[serialization](https://hazelcast.com/glossary/serialization/)/[deserialization](https://hazelcast.com/glossary/deserialization/)).

If you have a lot of Functions and a lot of Python packages, you might want to
keep the imports in the global scope so that every function can use the same
imports. In that case, you can use the
[`Image.imports`](https://modal.com/docs/reference/modal.Image#imports) context manager:

```python
pandas_image = modal.Image.debian_slim().pip_install("pandas", "numpy")

with pandas_image.imports():
    import pandas as pd
    import numpy as np

@app.function(image=pandas_image)
def my_function():
    df = pd.DataFrame()
    ...
```

Because these imports happen before a new container processes its first input,
you can combine this decorator with [memory snapshots](https://modal.com/docs/guide/memory-snapshot)
to improve [cold start performance](https://modal.com/docs/guide/cold-start#share-initialization-work-across-cold-starts-with-memory-snapshots)
for Functions that frequently scale from zero.

## Install system packages with `.apt_install`

You can install Linux packages with the [`apt` package manager](https://www.debian.org/doc/manuals/apt-guide/index.en.html)
using [`Image.apt_install`](https://modal.com/docs/reference/modal.Image#apt_install):

```python
image = modal.Image.debian_slim().apt_install("git", "curl")
```

## Set environment variables with `.env`

You can change the environment variables that your code sees
(in, e.g., [`os.environ`](https://docs.python.org/3/library/os.html#os.environ))
by passing a dictionary to [`Image.env`](https://modal.com/docs/refence/modal.Image#env):

```python
image = modal.Image.debian_slim().env({"PORT": "6443"})
```

Environment variable names and values must be strings.

## Run shell commands with `.run_commands`

You can supply shell commands that should be executed when building the
Image to [`Image.run_commands`](https://modal.com/docs/reference/modal.Image#run_commands):

```python
image_with_repo = (
    modal.Image.debian_slim().apt_install("git").run_commands(
        "git clone https://github.com/modal-labs/gpu-glossary"
    )
)
```

## Run a Python function during your build with `.run_function`

You can run Python code as a build step using the
[`Image.run_function`](https://modal.com/docs/reference/modal.Image#run_function) method.

For example, you can use this to download model parameters from Hugging Face into
your Image:

```python
import os

def download_models() -> None:
    import diffusers

    model_name = "segmind/small-sd"
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        model_name, use_auth_token=os.environ["HF_TOKEN"]
    )

hf_cache = modal.Volume.from_name("hf-cache")

image = (
    modal.Image.debian_slim()
        .pip_install("diffusers[torch]", "transformers", "ftfy", "accelerate")
        .run_function(
            download_models,
            secrets=[modal.Secret.from_name("huggingface-secret")],
            volumes={"/root/.cache/huggingface": hf_cache},
        )
)
```

For details on storing model weights on Modal, see
[this guide](https://modal.com/docs/guide/model-weights).

Essentially, this is equivalent to running a Modal Function and snapshotting the
resulting filesystem as a new Image. Any kwargs accepted by [`@app.function`](https://modal.com/docs/reference/modal.App#function)
([`Volume`s](https://modal.com/docs/guide/volumes), [`Secret`s](https://modal.com/docs/guide/secrets), specifications of
resources like [GPUs](https://modal.com/docs/guide/gpu)) can be supplied here.

Whenever you change other features of your Image, like the base Image or the
version of a Python package, the Image will automatically be rebuilt the next
time it is used. This is a bit more complicated when changing the contents of
functions. See the
[reference documentation](https://modal.com/docs/reference/modal.Image#run_function) for details.

## Attach GPUs during setup

If a step in the setup of your Image should be run on an instance with
a GPU (e.g., so that a package can query the GPU to set compilation flags), pass the
desired GPU type when defining that step:

```python
image = (
    modal.Image.debian_slim()
    .pip_install("bitsandbytes", gpu="H100")
)
```

## Use `mamba` instead of `pip` with `micromamba_install`

`pip` installs Python packages, but some Python workloads require the
coordinated installation of system packages as well. The `mamba` package manager
can install both. Modal provides a pre-built
[Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)
base image that makes it easy to work with `micromamba`:

```python
app = modal.App("bayes-pgm")

numpyro_pymc_image = (
    modal.Image.micromamba()
    .micromamba_install("pymc==5.10.4", "numpyro==0.13.2", channels=["conda-forge"])
)

@app.function(image=numpyro_pymc_image)
def sample():
    import pymc as pm
    import numpyro as np

    print(f"Running on PyMC v{pm.__version__} with JAX/numpyro v{np.__version__} backend")
    ...
```

## Image caching and rebuilds

Modal uses the definition of an Image to determine whether it needs to be
rebuilt. If the definition hasn't changed since the last time you ran or
deployed your App, the previous version will be pulled from the cache.

Images are cached per layer (i.e., per `Image` method call), and breaking
the cache on a single layer will cause cascading rebuilds for all subsequent
layers. You can shorten iteration cycles by defining frequently-changing
layers last so that the cached version of all other layers can be used.

In some cases, you may want to force an Image to rebuild, even if the
definition hasn't changed. You can do this by adding the `force_build=True`
argument to any of the Image building methods.

```python
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("slack-sdk", force_build=True)
    .run_commands("echo hi")
)
```

As in other cases where a layer's definition changes, both the `pip_install` and
`run_commands` layers will rebuild, but the `apt_install` will not. Remember to
remove `force_build=True` after you've rebuilt the Image, or it will
rebuild every time you run your code.

Alternatively, you can set the `MODAL_FORCE_BUILD` environment variable (e.g.
`MODAL_FORCE_BUILD=1 modal run ...`) to rebuild all images attached to your App.
But note that when you rebuild a base layer, the cache will be invalidated for _all_
Images that depend on it, and they will rebuild the next time you run or deploy
any App that uses that base. If you're debugging an issue with your Image, a better
option might be using `MODAL_IGNORE_CACHE=1`. This will rebuild the Image from the
top without breaking the Image cache or affecting subsequent builds.

## Image builder updates

Because changes to base images will cause cascading rebuilds, Modal is
conservative about updating the base definitions that we provide. But many
things are baked into these definitions, like the specific versions of the Image
OS, the included Python, and the Modal client dependencies.

We provide a separate mechanism for keeping base images up-to-date without
causing unpredictable rebuilds: the "Image Builder Version". This is a workspace
level-configuration that will be used for every Image built in your workspace.
We release a new Image Builder Version every few months but allow you to update
your workspace's configuration when convenient. After updating, your next
deployment will take longer, because your Images will rebuild. You may also
encounter problems, especially if your Image definition does not pin the version
of the third-party libraries that it installs (as your new Image will get the
latest version of these libraries, which may contain breaking changes).

You can set the Image Builder Version for your workspace by going to your
[workspace settings](https://modal.com/settings/image-config). This page also documents the
important updates in each version.

#### Using existing container images

# Using existing images

This guide walks you through how to use an existing container image as a Modal Image.

```python notest
sklearn_image = modal.Image.from_registry("huanjason/scikit-learn")
custom_image = modal.Image.from_dockerfile("./src/Dockerfile")
```

## Load an image from a public registry with `.from_registry`

To load an image from a public registry, just pass the image name, including any tags, to [`Image.from_registry`](https://modal.com/docs/reference/modal.Image#from_registry):

```python
sklearn_image = modal.Image.from_registry("huanjason/scikit-learn")

@app.function(image=sklearn_image)
def fit_knn():
    from sklearn.neighbors import KNeighborsClassifier
    ...
```

The `from_registry` method can load images from all public registries, such as
[Nvidia's `nvcr.io`](https://catalog.ngc.nvidia.com/containers),
[AWS ECR](https://aws.amazon.com/ecr/), and
[GitHub's `ghcr.io`](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).

You can further modify the image [just like any other Modal Image](https://modal.com/docs/guide/images):

```python continuation
data_science_image = sklearn_image.uv_pip_install("polars", "datasette")
```

You can use external images so long as

- The image is built for the
  [`linux/amd64` platform](https://unix.stackexchange.com/questions/53415/why-are-64-bit-distros-often-called-amd64)
- The image has a [compatible `ENTRYPOINT`](#entrypoint)

Additionally, to be used with a Modal Function, the image needs to have `python` and `pip`
installed and available on the `$PATH`.
If an existing image does not have either `python` or `pip` set up compatibly, you
can still use it. Just provide a version number as the `add_python` argument to
install a reproducible
[standalone build](https://github.com/indygreg/python-build-standalone)
of Python:

```python
ubuntu_image = modal.Image.from_registry("ubuntu:22.04", add_python="3.11")
valhalla_image = modal.Image.from_registry("gisops/valhalla:latest", add_python="3.12")
```

There are some additional restrictions for older versions of the Modal image builder.
Image builder version is set at a workspace level via the settings page [here](https://modal.com/settings/image-config).
See the migration guides on that page for details on any additional restrictions on images.

## Load images from private registries

You can also use images defined in private container registries on Modal.
The exact method depends on the registry you are using.

### Docker Hub (Private)

To pull container images from private Docker Hub repositories,
[create an access token](https://docs.docker.com/security/for-developers/access-tokens/)
with "Read-Only" permissions and use this token value and your Docker Hub
username to create a Modal [Secret](https://modal.com/docs/guide/secrets).

```
REGISTRY_USERNAME=my-dockerhub-username
REGISTRY_PASSWORD=dckr_pat_TS012345aaa67890bbbb1234ccc
```

Use this Secret with the
[`modal.Image.from_registry`](https://modal.com/docs/reference/modal.Image#from_registry) method.

### Elastic Container Registry (ECR)

You can pull images from your AWS ECR account by specifying the full image URI
as follows:

```python
import modal

aws_secret = modal.Secret.from_name("my-aws-secret")
image = (
    modal.Image.from_aws_ecr(
        "000000000000.dkr.ecr.us-east-1.amazonaws.com/my-private-registry:latest",
        secret=aws_secret,
    )
    .pip_install("torch", "huggingface")
)

app = modal.App(image=image)
```

As shown above, you also need to use a [Modal Secret](https://modal.com/docs/guide/secrets)
containing the environment variables `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, and `AWS_REGION`. The AWS IAM user account associated
with those keys must have access to the private registry you want to access.

Alternatively, you can use [OIDC token authentication](https://modal.com/docs/guide/oidc-integration#pull-images-from-aws-elastic-container-registry-ecr).

The user needs to have the following read-only policies:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": ["ecr:GetAuthorizationToken"],
      "Effect": "Allow",
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:GetRepositoryPolicy",
        "ecr:DescribeRepositories",
        "ecr:ListImages",
        "ecr:DescribeImages",
        "ecr:BatchGetImage",
        "ecr:GetLifecyclePolicy",
        "ecr:GetLifecyclePolicyPreview",
        "ecr:ListTagsForResource",
        "ecr:DescribeImageScanFindings"
      ],
      "Resource": ""
    }
  ]
}
```

You can use the IAM configuration above as a template for creating an IAM user.
You can then
[generate an access key](https://aws.amazon.com/premiumsupport/knowledge-center/create-access-key/)
and create a Modal Secret using the AWS integration option. Modal will use your
access keys to generate an ephemeral ECR token. That token is only used to pull
image layers at the time a new image is built. We don't store this token but
will cache the image once it has been pulled.

Images on ECR must be private and follow
[image configuration requirements](https://modal.com/docs/reference/modal.Image#from_aws_ecr).

### Google Artifact Registry and Google Container Registry

For further detail on how to pull images from Google's image registries, see
[`modal.Image.from_gcp_artifact_registry`](https://modal.com/docs/reference/modal.Image#from_gcp_artifact_registry).

## Bring your own image definition with `.from_dockerfile`

You can define an Image from an existing Dockerfile by passing its path to
[`Image.from_dockerfile`](https://modal.com/docs/reference/modal.Image#from_dockerfile):

```python
dockerfile_image = modal.Image.from_dockerfile("Dockerfile")

@app.function(image=dockerfile_image)
def fit():
    import sklearn
    ...
```

Note that you can still extend this Image using image builder methods!
See [the guide](https://modal.com/docs/guide/images) for details.

### Dockerfile command compatibility

Since Modal doesn't use Docker to build containers, we have our own
implementation of the
[Dockerfile specification](https://docs.docker.com/engine/reference/builder/).
Most Dockerfiles should work out of the box, but there are some differences to
be aware of.

First, a few minor Dockerfile commands and flags have not been implemented yet.
These include `ONBUILD`, `STOPSIGNAL`, and `VOLUME`.
Please reach out to us if your use case requires any of these.

Next, there are some command-specific things that may be useful when porting a
Dockerfile to Modal.

#### `ENTRYPOINT`

While the
[`ENTRYPOINT`](https://docs.docker.com/engine/reference/builder/#entrypoint)
command is supported, there is an additional constraint to the entrypoint script
provided: when used with a Modal Function, it must also `exec` the arguments passed to it at some point.
This is so the Modal Function runtime's Python entrypoint can run after your own. Most entrypoint
scripts in Docker containers are wrappers over other scripts, so this is likely
already the case.

If you wish to write your own entrypoint script, you can use the following as a
template:

```bash
#!/usr/bin/env bash

# Your custom startup commands here.

exec "$@" # Runs the command passed to the entrypoint script.
```

If the above file is saved as `/usr/bin/my_entrypoint.sh` in your container,
then you can register it as an entrypoint with
`ENTRYPOINT ["/usr/bin/my_entrypoint.sh"]` in your Dockerfile, or with
[`entrypoint`](https://modal.com/docs/reference/modal.Image#entrypoint) as an
Image build step.

```python
import modal

image = (
    modal.Image.debian_slim()
    .pip_install("foo")
    .entrypoint(["/usr/bin/my_entrypoint.sh"])
)
```

#### `ENV`

We currently don't support default values in
[interpolations](https://docs.docker.com/compose/compose-file/12-interpolation/),
such as `${VAR:-default}`

#### Fast pull from registry

# Fast pull from registry

The performance of pulling public and private images from registries into Modal
can be significantly improved by adopting the [eStargz](https://github.com/containerd/stargz-snapshotter/blob/main/docs/estargz.md) compression format.

By applying eStargz compression during your image build and push, Modal will be much
more efficient at pulling down your image from the registry.

## How to use estargz

If you have [Buildkit](https://docs.docker.com/build/buildkit/) version greater than `0.10.0`, adopting `estargz` is as simple as
adding some flags to your `docker buildx build` command:

- `type=registry` flag will instruct BuildKit to push the image after building.
  - If you do not push the image from immediately after build and instead attempt to push it later with docker push, the image will be converted to a standard gzip image.
- `compression=estargz` specifies that we are using the [eStargz](https://github.com/containerd/stargz-snapshotter/blob/main/docs/estargz.md) compression format.
- `oci-mediatypes=true` specifies that we are using the OCI media types, which is required for eStargz.
- `force-compression=true` will recompress the entire image and convert the base image to eStargz if it is not already.

```bash
docker buildx build --tag "<registry>/<namespace>/<repo>:<version>" \
--output type=registry,compression=estargz,force-compression=true,oci-mediatypes=true \
.
```

Then reference the container image as normal in your Modal code.

```python notest
app = modal.App(
    "example-estargz-pull",
    image=modal.Image.from_registry(
        "public.ecr.aws/modal/estargz-example-images:text-generation-v1-esgz"
    )
)
```

At build time you should see the eStargz-enabled puller activate:

```
Building image im-TinABCTIf12345ydEwTXYZ

=> Step 0: FROM public.ecr.aws/modal/estargz-example-images:text-generation-v1-esgz
Using estargz to speed up image pull (index loaded in 1.86s)...
Progress: 10% complete... (1.11s elapsed)
Progress: 20% complete... (3.10s elapsed)
Progress: 30% complete... (4.18s elapsed)
Progress: 40% complete... (4.76s elapsed)
Progress: 50% complete... (5.51s elapsed)
Progress: 62% complete... (6.17s elapsed)
Progress: 74% complete... (6.99s elapsed)
Progress: 81% complete... (7.23s elapsed)
Progress: 99% complete... (8.90s elapsed)
Progress: 100% complete... (8.90s elapsed)
Copying image...
Copied image in 5.81s
```

## Supported registries

Currently, Modal supports fast estargz pulling images with the following registries:

- AWS Elastic Container Registry (ECR)
- Docker Hub (docker.io)
- Google Artifact Registry (gcr.io, pkg.dev)

We are working on adding support for GitHub Container Registry (ghcr.io).

### GPUs and other resources

#### GPU acceleration

# GPU acceleration

Modal makes it easy to run your code on [GPUs](https://modal.com/gpu-glossary/readme).

## Quickstart

Here's a simple example of a Function running on an A100 in Modal:

```python
import modal

image = modal.Image.debian_slim().pip_install("torch")
app = modal.App(image=image)

@app.function(gpu="A100")
def run():
    import torch

    assert torch.cuda.is_available()
```

## Specifying GPU type

You can pick a specific GPU type for your Function via the `gpu` argument.
Modal supports the following values for this parameter:

- `T4`
- `L4`
- `A10`
- `A100`
- `A100-40GB`
- `A100-80GB`
- `L40S`
- `H100`/`H100!`
- `H200`
- `B200`

For instance, to use a B200, you can use `@app.function(gpu="B200")`.

Refer to our [pricing page](https://modal.com/pricing) for the latest pricing on each GPU type.

## Specifying GPU count

You can specify more than 1 GPU per container by appending `:n` to the GPU
argument. For instance, to run a Function with eight H100s:

```python

@app.function(gpu="H100:8")
def run_llama_405b_fp8():
    ...
```

Currently B200, H200, H100, A100, L4, T4 and L40S instances support up to 8 GPUs (up to 1,536 GB GPU RAM),
and A10 instances support up to 4 GPUs (up to 96 GB GPU RAM). Note that requesting
more than 2 GPUs per container will usually result in larger wait times. These
GPUs are always attached to the same physical machine.

## Picking a GPU

For running, rather than training, neural networks, we recommend starting off
with the [L40S](https://resources.nvidia.com/en-us-l40s/l40s-datasheet-28413),
which offers an excellent trade-off of cost and performance and 48 GB of GPU
RAM for storing model weights and activations.

For more on how to pick a GPU for use with neural networks like LLaMA or Stable
Diffusion, and for tips on how to make that GPU go brrr, check out
[Tim Dettemers' blog post](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/)
or the
[Full Stack Deep Learning page on Cloud GPUs](https://fullstackdeeplearning.com/cloud-gpus/).

## B200 GPUs

Modal's most powerful GPUs are the [B200s](https://www.nvidia.com/en-us/data-center/dgx-b200/),
NVIDIA's flagship data center chip for the Blackwell [architecture](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor-architecture).

To request a B200, set the `gpu` argument to `"B200"`

```python
@app.function(gpu="B200:8")
def run_deepseek():
    ...
```

Check out [this example](https://modal.com/docs/examples/vllm_inference) to see how you can use B200s to max out vLLM serving performance for LLaMA 3.1-8B.

Before you jump for the most powerful (and so most expensive) GPU, make sure you
understand where the bottlenecks are in your computations. For example, running
language models with small batch sizes (e.g. one prompt at a time) results in a
[bottleneck on memory, not arithmetic](https://kipp.ly/transformer-inference-arithmetic/).
Since arithmetic throughput has risen faster than memory throughput in recent
hardware generations, speedups for memory-bound GPU jobs are not as extreme and
may not be worth the extra cost.

## H200 and H100 GPUs

[H200s](https://www.nvidia.com/en-us/data-center/h200/) and [H100s](https://www.nvidia.com/en-us/data-center/h100/) are the previous
generation of top-of-the-line data center chips from NVIDIA, based on the Hopper [architecture](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor-architecture).
These GPUs have better software support than do Blackwell GPUs (e.g. popular libraries include pre-compiled kernels for Hopper, but not Blackwell),
and they often get the job done at a competitive cost, so they are a common choice of accelerator, on and off Modal.

All H100 GPUs on the Modal platform are of the SXM variant, as can be verified by examining the
[power draw](https://modal.com/docs/guide/gpu-metrics) in the dashboard or with `nvidia-smi`.

### Automatic upgrades to H200s

Modal may automatically upgrade a `gpu="H100"` request to run on an H200.
This automatic upgrade does _not_ change the cost of the GPU.

Kernels [compatible](https://modal.com/gpu-glossary/device-software/compute-capability) with H200s are also compatible with H100s,
so your code will still run, just faster, so long as it doesn't make strict assumptions about memory capacity.
An H200’s [HBM3e memory](https://modal.com/gpu-glossary/device-hardware/gpu-ram)
has a capacity of 141 GB and a bandwidth of 4.8TB/s, 1.75x larger and 1.4x faster than an NVIDIA H100 with HBM3.

In cases where an automatic upgrade to H200 would not be helpful (for instance, benchmarking) you can pass
`gpu=H100!` to avoid it.

## A100 GPUs

[A100s](https://www.nvidia.com/en-us/data-center/a100/) are based on NVIDIA's Ampere [architecture](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor-architecture).
Modal offers two versions of the A100: one with 40 GB of RAM and another with 80 GB of RAM.

To request an A100 with 40 GB of [GPU memory](https://modal.com/gpu-glossary/device-hardware/gpu-ram), use `gpu="A100"`:

```python
@app.function(gpu="A100")
def qwen_7b():
    ...
```

Modal may automatically upgrade a `gpu="A100"` request to run on an 80 GB A100.
This automatic upgrade does _not_ change the cost of the GPU.

You can specifically request a 40GB A100 with the string `A100-40GB`.
To specifically request an 80 GB A100, use the string `A100-80GB`:

```python
@app.function(gpu="A100-80GB")
def llama_70b_fp8():
    ...
```

## GPU fallbacks

Modal allows specifying a list of possible GPU types, suitable for Functions that are
compatible with multiple options. Modal respects the ordering of this list and
will try to allocate the most preferred GPU type before falling back to less
preferred ones.

```python
@app.function(gpu=["H100", "A100-40GB:2"])
def run_on_80gb():
    ...
```

See [this example](https://modal.com/docs/examples/gpu_fallbacks) for more detail.

## Multi GPU training

Modal currently supports multi-GPU training on a single node, with multi-node training in closed beta ([contact us](https://modal.com/slack) for access).
Depending on which framework you are using, you may need to use different techniques to train on multiple GPUs.

If the framework re-executes the entrypoint of the Python process (like [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/index.html)) you need to either set the strategy to `ddp_spawn` or `ddp_notebook` if you wish to invoke the training directly. Another option is to run the training script as a subprocess instead.

```python
@app.function(gpu="A100:2")
def run():
    import subprocess
    import sys
    subprocess.run(
        ["python", "train.py"],
        stdout=sys.stdout, stderr=sys.stderr,
        check=True,
    )
```

## Examples and more resources

For more information about GPUs in general, check out our [GPU Glossary](https://modal.com/gpu-glossary/readme).

Or take a look some examples of Modal apps using GPUs:

- [Fine-tune a character LoRA for your pet](https://modal.com/docs/examples/dreambooth_app)
- [Fast LLM inference with vLLM](https://modal.com/docs/examples/vllm_inference)
- [Stable Diffusion with a CLI, API, and web UI](https://modal.com/docs/examples/stable_diffusion_cli)
- [Rendering Blender videos](https://modal.com/docs/examples/blender_video)

#### Using CUDA on Modal

# Using CUDA on Modal

Modal makes it easy to accelerate your workloads with datacenter-grade NVIDIA GPUs.

To take advantage of the hardware, you need to use matching software: the CUDA stack.
This guide explains the components of that stack and how to install them on Modal.
For more on which GPUs are available on Modal and how to choose a GPU for your use case,
see [this guide](https://modal.com/docs/guide/gpu). For a deep dive on both the
[GPU hardware](https://modal.com/gpu-glossary/device-hardware) and [software](https://modal.com/gpu-glossary/device-software)
and for even more detail on [the CUDA stack](https://modal.com/gpu-glossary/host-software/),
see our [GPU Glossary](https://modal.com/gpu-glossary/readme).

Here's the tl;dr:

- The [NVIDIA Accelerated Graphics Driver for Linux-x86_64](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#driver-installation), version 575.57.08,
  and [CUDA Driver API](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/index.html), version 12.8, are already installed.
  You can call `nvidia-smi` or run compiled CUDA programs from any Modal Function with access to a GPU.
- That means you can install many popular libraries like `torch` that bundle their other CUDA dependencies [with a simple `pip_install`](#install-gpu-accelerated-torch-and-transformers-with-pip_install).
- For bleeding-edge libraries like `flash-attn`, you may need to install CUDA dependencies manually.
  To make your life easier, [use an existing image](#for-more-complex-setups-use-an-officially-supported-cuda-image).

## What is CUDA?

When someone refers to "installing CUDA" or "using CUDA",
they are referring not to a library, but to a
[stack](https://modal.com/gpu-glossary/host-software/cuda-software-platform) with multiple layers.
Your application code (and its dependencies) can interact
with the stack at different levels.

![The CUDA stack](../../assets/docs/cuda-stack-diagram.png)

This leads to a lot of confusion. To help clear that up, the following sections explain each component in detail.

### Level 0: Kernel-mode driver components

At the lowest level are the [_kernel-mode driver components_](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#nvidia-open-gpu-kernel-modules).
The Linux kernel is essentially a single program operating the entire machine and all of its hardware.
To add hardware to the machine, this program is extended by loading new modules into it.
These components communicate directly with hardware -- in this case the GPU.

Because they are kernel modules, these driver components are tightly integrated with the host operating system
that runs your containerized Modal Functions and are not something you can inspect or change yourself.

### Level 1: User-mode driver API

All action in Linux that doesn't occur in the kernel occurs in [user space](https://en.wikipedia.org/wiki/User_space).
To talk to the kernel drivers from our user space programs, we need _user-mode driver components_.

Most prominently, that includes:

- the [CUDA Driver API](https://modal.com/gpu-glossary/host-software/cuda-driver-api),
  a [shared object](https://en.wikipedia.org/wiki/Shared_library) called `libcuda.so`.
  This object exposes functions like [`cuMemAlloc`](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467),
  for allocating GPU memory.
- the [NVIDIA management library](https://developer.nvidia.com/management-library-nvml), `libnvidia-ml.so`, and its command line interface [`nvidia-smi`](https://developer.nvidia.com/system-management-interface).
  You can use these tools to check the status of the system's GPU(s).

These components are installed on all Modal machines with access to GPUs.
Because they are user-level components, you can use them directly:

```python runner:ModalRunner
import modal

app = modal.App()

@app.function(gpu="any")
def check_nvidia_smi():
    import subprocess
    output = subprocess.check_output(["nvidia-smi"], text=True)
    assert "Driver Version:" in output
    assert "CUDA Version:" in output
    print(output)
    return output
```

### Level 2: CUDA Toolkit

Wrapping the CUDA Driver API is the [CUDA Runtime API](https://modal.com/gpu-glossary/host-software/cuda-runtime-api), the `libcudart.so` shared library.
This API includes functions like [`cudaLaunchKernel`](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g7656391f2e52f569214adbfc19689eb3)
and is more commonly used in CUDA programs (see [this HackerNews comment](https://news.ycombinator.com/item?id=20616385) for color commentary on why).
This shared library is _not_ installed by default on Modal.

The CUDA Runtime API is generally installed as part of the larger [NVIDIA CUDA Toolkit](https://docs.nvidia.com/cuda/index.html),
which includes the [NVIDIA CUDA compiler driver](https://modal.com/gpu-glossary/host-software/nvcc) (`nvcc`) and its toolchain
and a number of [useful goodies](https://modal.com/gpu-glossary/host-software/cuda-binary-utilities) for writing and debugging CUDA programs (`cuobjdump`, `cudnn`, profilers, etc.).

Contemporary GPU-accelerated machine learning workloads like LLM inference frequently make use of many components of the CUDA Toolkit,
such as the run-time compilation library [`nvrtc`](https://docs.nvidia.com/cuda/archive/12.8.0/nvrtc/index.html).

So why aren't these components installed along with the drivers?
A compiled CUDA program can run without the CUDA Runtime API installed on the system,
by [statically linking](https://en.wikipedia.org/wiki/Static_library) the CUDA Runtime API into the program binary,
though this is fairly uncommon for CUDA-accelerated Python programs.
Additionally, older versions of these components are needed for some applications
and some application deployments even use several versions at once.
Both patterns are compatible with the host machine driver provided on Modal.

## Install GPU-accelerated `torch` and `transformers` with `pip_install`

The components of the CUDA Toolkit can be installed via `pip`,
via PyPI packages like [`nvidia-cuda-runtime-cu12`](https://pypi.org/project/nvidia-cuda-runtime-cu12/)
and [`nvidia-cuda-nvrtc-cu12`](https://pypi.org/project/nvidia-cuda-nvrtc-cu12/).
These components are listed as dependencies of some popular GPU-accelerated Python libraries, like `torch`.

Because Modal already includes the lower parts of the CUDA stack, you can install these libraries
with [the `pip_install` method of `modal.Image`](https://modal.com/docs/guide/images#add-python-packages-with-pip_install), just like any other Python library:

```python
image = modal.Image.debian_slim().pip_install("torch")

@app.function(gpu="any", image=image)
def run_torch():
    import torch
    has_cuda = torch.cuda.is_available()
    print(f"It is {has_cuda} that torch can access CUDA")
    return has_cuda
```

Many libraries for running open-weights models, like `transformers` and `vllm`,
use `torch` under the hood and so can be installed in the same way:

```python
image = modal.Image.debian_slim().pip_install("transformers[torch]")
image = image.apt_install("ffmpeg")  # for audio processing

@app.function(gpu="any", image=image)
def run_transformers():
    from transformers import pipeline
    transcriber = pipeline(model="openai/whisper-tiny.en", device="cuda")
    result = transcriber("https://modal-cdn.com/mlk.flac")
    print(result["text"])  # I have a dream that one day this nation will rise up live out the true meaning of its creed
```

## For more complex setups, use an officially-supported CUDA image

The disadvantage of installing the CUDA stack via `pip` is that
many other libraries that depend on its components being installed as normal system packages cannot find them.

For these cases, we recommend you use an image that already has the full CUDA stack installed as system packages
and all environment variables set correctly, like the [`nvidia/cuda:*-devel-*` images on Docker Hub](https://hub.docker.com/r/nvidia/cuda).

[TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/overview.html) is an inference engine that accelerates and optimizes performance for the large language models. It requires the full CUDA toolkit for installation.

```python
cuda_version = "12.8.1"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
HF_CACHE_PATH = "/cache"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .entrypoint([])  # remove verbose logging by base image on entry
    .apt_install("libopenmpi-dev")  # required for tensorrt
    .pip_install("tensorrt-llm==0.19.0", "pynvml", extra_index_url="https://pypi.nvidia.com")
    .pip_install("hf-transfer", "huggingface_hub[hf_xet]")
    .env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1", "PMIX_MCA_gds": "hash"})
)

app = modal.App("tensorrt-llm", image=image)
hf_cache_volume = modal.Volume.from_name("hf_cache_tensorrt", create_if_missing=True)

@app.function(gpu="A10G", volumes={HF_CACHE_PATH: hf_cache_volume})
def run_tiny_model():
    from tensorrt_llm import LLM, SamplingParams

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    output = llm.generate("The capital of France is", sampling_params)
    print(f"Generated text: {output.outputs[0].text}")
    return output.outputs[0].text
```

Make sure to choose a version of CUDA that is no greater than the version provided by the host machine.
Older minor (`12.*`) versions are guaranteed to be compatible with the host machine's driver,
but older major (`11.*`, `10.*`, etc.) versions may not be.

## What next?

For more on accessing and choosing GPUs on Modal, check out [this guide](https://modal.com/docs/guide/gpu).
To dive deep on GPU internals, check out our [GPU Glossary](https://modal.com/gpu-glossary/readme).

To see these installation patterns in action, check out these examples:

- [Fast LLM inference with vLLM](https://modal.com/docs/examples/vllm_inference)
- [Finetune a character LoRA for your pet](https://modal.com/docs/examples/diffusers_lora_finetune)
- [Optimized Flux inference](https://modal.com/docs/examples/flux)

#### Reserving CPU and memory

# Reserving CPU and memory

Each Modal container has a default reservation of 0.125 CPU cores and 128 MiB of memory.
Containers can exceed this minimum if the worker has available CPU or memory.
You can also guarantee access to more resources by requesting a higher reservation.

## CPU cores

If you have code that must run on a larger number of cores, you can
request that using the `cpu` argument. This allows you to specify a
floating-point number of CPU cores:

```python
import modal

app = modal.App()

@app.function(cpu=8.0)
def my_function():
    # code here will have access to at least 8.0 cores
    ...
```

Note that this value corresponds to physical cores, not vCPUs.

Modal also will set several environment variables that control multi-threading
behavior in linear algebra libraries (e.g., `OPENBLAS_NUM_THREADS`,
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`) based on your CPU reservation.

## Memory

If you have code that needs more guaranteed memory, you can request it using the
`memory` argument. This expects an integer number of megabytes:

```python
import modal

app = modal.App()

@app.function(memory=32768)
def my_function():
    # code here will have access to at least 32 GiB of RAM
    ...
```

## How much can I request?

For both CPU and memory, a maximum is enforced at Function creation time to
ensure your containers can be scheduled for execution. Requests exceeding the
maximum will be rejected with an
[`InvalidError`](https://modal.com/docs/reference/modal.exception#modalexceptioninvaliderror).

## Billing

For CPU and memory, you'll be charged based on whichever is higher: your reservation or actual usage.

Disk requests are billed by increasing the memory request at a 20:1 ratio. For example, requesting 500 GiB of disk will increase the memory request to 25 GiB, if it is not already set higher.

## Resource limits

### CPU limits

Modal containers have a default soft CPU limit that is set at 16 physical cores above the CPU request.
Given that the default CPU request is 0.125 cores, the default soft CPU limit is 16.125 cores.
Above this limit, the host will begin to throttle the CPU usage of the container.

You can alternatively set the CPU limit explicitly:

```python
cpu_request = 1.0
cpu_limit = 4.0
@app.function(cpu=(cpu_request, cpu_limit))
def f():
    ...
```

### Memory limits

Modal containers can have a hard memory limit which will 'Out of Memory' (OOM) kill
containers which attempt to exceed the limit. This functionality is useful when a process
has a serious memory leak. You can set the limit and have the container killed to avoid paying
for the leaked GBs of memory.

Specify this limit using the [`memory` parameter](https://modal.com/docs/reference/modal.App#function):

```python
mem_request = 1024
mem_limit = 2048
@app.function(
    memory=(mem_request, mem_limit),
)
def f():
    ...
```

### Disk limits

Running Modal containers have access to many GBs of SSD disk, but the amount
of writes is limited by:

1. The size of the underlying worker's SSD disk capacity
2. A per-container disk quota that is set in the 100s of GBs.

Hitting either limit will cause the container's disk writes to be rejected, which
typically manifests as an `OSError`.

Increased disk sizes can be requested with the [`ephemeral_disk` parameter](https://modal.com/docs/reference/modal.App#function). The maximum
disk size is 3.0 TiB (3,145,728 MiB). Larger disks are intended to be used for [dataset processing](https://modal.com/docs/guide/dataset-ingestion).

### Scaling out

#### Scaling out

# Scaling out

Modal makes it trivially easy to scale compute across thousands of containers.
You won't have to worry about your App crashing if it goes viral or need to wait
a long time for your batch jobs to complete.

For the the most part, scaling out will happen automatically, and you won't need
to think about it. But it can be helpful to understand how Modal's autoscaler
works and how you can control its behavior when you need finer control.

## How does autoscaling work on Modal?

Every Modal Function corresponds to an autoscaling pool of containers. The size
of the pool is managed by Modal's autoscaler. The autoscaler will spin up new
containers when there is no capacity available for new inputs, and it will spin
down containers when resources are idling. By default, Modal Functions will
scale to zero when there are no inputs to process.

Autoscaling decisions are made quickly and frequently so that your batch jobs
can ramp up fast and your deployed Apps can respond to any sudden changes in
traffic.

## Configuring autoscaling behavior

Modal exposes a few settings that allow you to configure the autoscaler's
behavior. These settings can be passed to the `@app.function` or `@app.cls`
decorators:

- `max_containers`: The upper limit on containers for the specific Function.
- `min_containers`: The minimum number of containers that should be kept warm,
  even when the Function is inactive.
- `buffer_containers`: The size of the buffer to maintain while the Function is
  active, so that additional inputs will not need to queue for a new container.
- `scaledown_window`: The maximum duration (in seconds) that individual
  containers can remain idle when scaling down.

In general, these settings allow you to trade off cost and latency. Maintaining
a larger warm pool or idle buffer will increase costs but reduce the chance that
inputs will need to wait for a new container to start.

Similarly, a longer scaledown window will let containers idle for longer, which
might help avoid unnecessary churn for Apps that receive regular but infrequent
inputs. Note that containers may not wait for the entire scaledown window before
shutting down if the App is substantially overprovisioned.

## Dynamic autoscaler updates

It's also possible to update the autoscaler settings dynamically (i.e., without redeploying
the App) using the [`Function.update_autoscaler()`](https://modal.com/docs/reference/modal.Function#update_autoscaler)
method:

```python notest
f = modal.Function.from_name("my-app", "f")
f.update_autoscaler(max_containers=100)
```

The autoscaler settings will revert to the configuration in the function
decorator the next time you deploy the App. Or they can be overridden by
further dynamic updates:

```python notest
f.update_autoscaler(min_containers=2, max_containers=10)
f.update_autoscaler(min_containers=4)  # max_containers=10 will still be in effect
```

A common pattern is to run this method in a [scheduled function](https://modal.com/docs/guide/cron)
that adjusts the size of the warm pool (or container buffer) based on the time of day:

```python
@app.function()
def inference_server():
    ...

@app.function(schedule=modal.Cron("0 6 * * *", timezone="America/New_York"))
def increase_warm_pool():
    inference_server.update_autoscaler(min_containers=4)

@app.function(schedule=modal.Cron("0 22 * * *", timezone="America/New_York"))
def decrease_warm_pool():
    inference_server.update_autoscaler(min_containers=0)
```

When you have a [`modal.Cls`](https://modal.com/docs/reference/modal.Cls), `update_autoscaler`
is a method on an _instance_ and will control the autoscaling behavior of
containers serving the Function with that specific set of parameters:

```python notest
MyClass = modal.Cls.from_name("my-app", "MyClass")
obj = MyClass(model_version="3.5")
obj.update_autoscaler(buffer_containers=2)  # type: ignore
```

Note that it's necessary to disable type checking on this line, because the
object will appear as an instance of the class that you defined rather than the
Modal wrapper type.

## Parallel execution of inputs

If your code is running the same function repeatedly with different independent
inputs (e.g., a grid search), the easiest way to increase performance is to run
those function calls in parallel using Modal's
[`Function.map()`](https://modal.com/docs/reference/modal.Function#map) method.

Here is an example if we had a function `evaluate_model` that takes a single
argument:

```python
import modal

app = modal.App()

@app.function()
def evaluate_model(x):
    ...

@app.local_entrypoint()
def main():
    inputs = list(range(100))
    for result in evaluate_model.map(inputs):  # runs many inputs in parallel
        ...
```

In this example, `evaluate_model` will be called with each of the 100 inputs
(the numbers 0 - 99 in this case) roughly in parallel and the results are
returned as an iterable with the results ordered in the same way as the inputs.

### Exceptions

By default, if any of the function calls raises an exception, the exception will
be propagated. To treat exceptions as successful results and aggregate them in
the results list, pass in
[`return_exceptions=True`](https://modal.com/docs/reference/modal.Function#map).

```python
@app.function()
def my_func(a):
    if a == 2:
        raise Exception("ohno")
    return a ** 2

@app.local_entrypoint()
def main():
    print(list(my_func.map(range(3), return_exceptions=True, wrap_returned_exceptions=False)))
    # [0, 1, Exception('ohno'))]
```

Note: prior to version 1.0.5, the returned exceptions inadvertently leaked an internal
wrapper type (`modal.exceptions.UserCodeException`). To avoid breaking any user code that
was checking exception types, we're taking a gradual approach to fixing this bug. Adding
`wrap_returned_exceptions=False` will opt-in to the future default behavior and return the
underlying exception type without a wrapper.

### Starmap

If your function takes multiple variable arguments, you can either use
[`Function.map()`](https://modal.com/docs/reference/modal.Function#map) with one input iterator
per argument, or [`Function.starmap()`](https://modal.com/docs/reference/modal.Function#starmap)
with a single input iterator containing sequences (like tuples) that can be
spread over the arguments. This works similarly to Python's built in `map` and
`itertools.starmap`.

```python
@app.function()
def my_func(a, b):
    return a + b

@app.local_entrypoint()
def main():
    assert list(my_func.starmap([(1, 2), (3, 4)])) == [3, 7]
```

### Gotchas

Note that `.map()` is a method on the modal function object itself, so you don't
explicitly _call_ the function.

Incorrect usage:

```python notest
results = evaluate_model(inputs).map()
```

Modal's map is also not the same as using Python's builtin `map()`. While the
following will technically work, it will execute all inputs in sequence rather
than in parallel.

Incorrect usage:

```python notest
results = map(evaluate_model, inputs)
```

## Asynchronous usage

All Modal APIs are available in both blocking and asynchronous variants. If you
are comfortable with asynchronous programming, you can use it to create
arbitrary parallel execution patterns, with the added benefit that any Modal
functions will be executed remotely. See the [async guide](https://modal.com/docs/guide/async) or
the examples for more information about asynchronous usage.

## GPU acceleration

Sometimes you can speed up your applications by utilizing GPU acceleration. See
the [gpu section](https://modal.com/docs/guide/gpu) for more information.

## Scaling Limits

Modal enforces the following limits for every function:

- 2,000 pending inputs (inputs that haven't been assigned to a container yet)
- 25,000 total inputs (which include both running and pending inputs)

For inputs created with `.spawn()` for async jobs, Modal allows up to 1 million pending inputs instead of 2,000.

If you try to create more inputs and exceed these limits, you'll receive a `Resource Exhausted` error, and you should retry your request later. If you need higher limits, please reach out!

Additionally, each `.map()` invocation can process at most 1000 inputs concurrently.

#### Input concurrency

# Input concurrency

As traffic to your application increases, Modal will automatically scale up the
number of containers running your Function:

<div class="flex justify-center"></div>

By default, each container will be assigned one input at a time. Autoscaling
across containers allows your Function to process inputs in parallel. This is
ideal when the operations performed by your Function are CPU-bound.

For some workloads, though, it is inefficient for containers to process inputs
one-by-one. Modal supports these workloads with its _input concurrency_ feature,
which allows individual containers to process multiple inputs at the same time:

<div class="flex justify-center"></div>

When used effectively, input concurrency can reduce latency and lower costs.

## Use cases

Input concurrency can be especially effective for workloads that are primarily
I/O-bound, e.g.:

- Querying a database
- Making external API requests
- Making remote calls to other Modal Functions

For such workloads, individual containers may be able to concurrently process
large numbers of inputs with minimal additional latency. This means that your
Modal application will be more efficient overall, as it won't need to scale
containers up and down as traffic ebbs and flows.

Another use case is to leverage _continuous batching_ on GPU-accelerated
containers. Frameworks such as [vLLM](https://modal.com/docs/examples/vllm_inference) can
achieve the benefits of batching across multiple inputs even when those
inputs do not arrive simultaneously (because new batches are formed for each
forward pass of the model).

Note that for CPU-bound workloads, input concurrency will likely not be as
effective (or will even be counterproductive), and you may want to use
Modal's [_dynamic batching_ feature](https://modal.com/docs/guide/dynamic-batching) instead.

## Enabling input concurrency

To enable input concurrency, add the `@modal.concurrent` decorator:

```python
@app.function()
@modal.concurrent(max_inputs=100)
def my_function(input: str):
    ...

```

When using the class pattern, the decorator should be applied at the level of
the _class_, not on individual methods:

```python
@app.cls()
@modal.concurrent(max_inputs=100)
class MyCls:

    @modal.method()
    def my_method(self, input: str):
        ...
```

Because all methods on a class will be served by the same containers, a class
with input concurrency enabled will concurrently run distinct methods in
addition to multiple inputs for the same method.

**Note:** The `@modal.concurrent` decorator was added in v0.73.148 of the Modal
Python SDK. Input concurrency could previously be enabled by setting the
`allow_concurrent_inputs` parameter on the `@app.function` decorator.

## Setting a concurrency target

When using the `@modal.concurrent` decorator, you must always configure the
maximum number of inputs that each container will concurrently process. If
demand exceeds this limit, Modal will automatically scale up more containers.

Additional inputs may need to queue up while these additional containers cold
start. To help avoid degraded latency during scaleup, the `@modal.concurrent`
decorator has a separate `target_inputs` parameter. When set, Modal's autoscaler
will aim for this target as it provisions resources. If demand increases faster
than new containers can spin up, the active containers will be allowed to burst
above the target up to the `max_inputs` limit:

```python
@app.function()
@modal.concurrent(max_inputs=120, target_inputs=100)  # Allow a 20% burst
def my_function(input: str):
    ...
```

It may take some experimentation to find the right settings for these parameters
in your particular application. Our suggestion is to set the `target_inputs`
based on your desired latency and the `max_inputs` based on resource constraints
(i.e., to avoid GPU OOM). You may also consider the relative latency cost of
scaling up a new container versus overloading the existing containers.

## Concurrency mechanisms

Modal uses different concurrency mechanisms to execute your Function depending
on whether it is defined as synchronous or asynchronous. Each mechanism imposes
certain requirements on the Function implementation. Input concurrency is an
advanced feature, and it's important to make sure that your implementation
complies with these requirements to avoid unexpected behavior.

For synchronous Functions, Modal will execute concurrent inputs on separate
threads. _This means that the Function implementation must be thread-safe._

```python
# Each container can execute up to 10 inputs in separate threads
@app.function()
@modal.concurrent(max_inputs=10)
def sleep_sync():
    # Function must be thread-safe
    time.sleep(1)
```

For asynchronous Functions, Modal will execute concurrent inputs using
separate `asyncio` tasks on a single thread. This does not require thread
safety, but it does mean that the Function needs to participate in
collaborative multitasking (i.e., it should not block the event loop).

```python
# Each container can execute up to 10 inputs with separate async tasks
@app.function()
@modal.concurrent(max_inputs=10)
async def sleep_async():
    # Function must not block the event loop
    await asyncio.sleep(1)
```

## Gotchas

Input concurrency is a powerful feature, but there are a few caveats that can
be useful to be aware of before adopting it.

### Input cancellations

Synchronous and asynchronous Functions handle input cancellations differently.
Modal will raise a `modal.exception.InputCancellation` exception in synchronous
Functions and an `asyncio.CancelledError` in asynchronous Functions.

When using input concurrency with a synchronous Function, a single input
cancellation will terminate the entire container. If your workflow depends on
graceful input cancellations, we recommend using an asynchronous
implementation.

### Concurrent logging

The separate threads or tasks that are executing the concurrent inputs will
write any logs to the same stream. This makes it difficult to associate logs
with a specific input, and filtering for a specific function call in Modal's web
dashboard will show logs for all inputs running at the same time.

To work around this, we recommend including a unique identifier in the messages
you log (either your own identifier or the `modal.current_input_id()`) so that
you can use the search functionality to surface logs for a specific input:

```python
@app.function()
@modal.concurrent(max_inputs=10)
async def better_concurrent_logging(x: int):
    logger.info(f"{modal.current_input_id()}: Starting work with {x}")
```

#### Batch processing

# Batch Processing

Modal is optimized for large-scale batch processing, allowing functions to scale to thousands of parallel containers with zero additional configuration. Function calls can be submitted asynchronously for background execution, eliminating the need to wait for jobs to finish or tune resource allocation.

This guide covers Modal's batch processing capabilities, from basic invocation to integration with existing pipelines.

## Background Execution with `.spawn_map`

The fastest way to submit multiple jobs for asynchronous processing is by invoking a function with `.spawn_map`. When combined with the [`--detach`](https://modal.com/docs/reference/cli/run) flag, your App continues running until all jobs are completed.

Here's an example of submitting 100,000 videos for parallel embedding. You can disconnect after submission, and the processing will continue to completion in the background:

```python
# Kick off asynchronous jobs with `modal run --detach batch_processing.py`
import modal

app = modal.App("batch-processing-example")
volume = modal.Volume.from_name("video-embeddings", create_if_missing=True)

@app.function(volumes={"/data": volume})
def embed_video(video_id: int):
    # Business logic:
    # - Load the video from the volume
    # - Embed the video
    # - Save the embedding to the volume
    ...

@app.local_entrypoint()
def main():
    embed_video.spawn_map(range(100_000))
```

This pattern works best for jobs that store results externally—for example, in a [Modal Volume](https://modal.com/docs/guide/volumes), [Cloud Bucket Mount](https://modal.com/docs/guide/cloud-bucket-mounts), or your own database\*.

_\* For database connections, consider using [Modal Proxy](https://modal.com/docs/guide/proxy-ips) to maintain a static IP across thousands of containers._

## Parallel Processing with `.map`

Using `.map` allows you to offload expensive computations to powerful machines while gathering results. This is particularly useful for pipeline steps with bursty resource demands. Modal handles all infrastructure provisioning and de-provisioning automatically.

Here's how to implement parallel video similarity queries as a single Modal function call:

```python
# Run jobs and collect results with `modal run gather.py`
import modal

app = modal.App("gather-results-example")

@app.function(gpu="L40S")
def compute_video_similarity(query: str, video_id: int) -> tuple[int, int]:
    # Embed video with GPU acceleration & compute similarity with query
    return video_id, score

@app.local_entrypoint()
def main():
    import itertools

    queries = itertools.repeat("Modal for batch processing")
    video_ids = range(100_000)

    for video_id, score in compute_video_similarity.map(queries, video_ids):
        # Process results (e.g., extract top 5 most similar videos)
        pass
```

This example runs `compute_video_similarity` on an autoscaling pool of L40S GPUs, returning scores to a local process for further processing.

## Integration with Existing Systems

The recommended way to use Modal Functions within your existing data pipeline is through [deployed function invocation](https://modal.com/docs/guide/trigger-deployed-functions). After deployment, you can call Modal functions from external systems:

```python
def external_function(inputs):
    compute_similarity = modal.Function.from_name(
        "gather-results-example",
        "compute_video_similarity"
    )
    for result in compute_similarity.map(inputs):
        # Process results
        pass
```

You can invoke Modal Functions from any Python context, gaining access to built-in observability, resource management, and GPU acceleration.

#### Job queues

# Job processing

Modal can be used as a scalable job queue to handle asynchronous tasks submitted
from a web app or any other Python application. This allows you to offload up to 1 million
long-running or resource-intensive tasks to Modal, while your main application
remains responsive.

## Creating jobs with .spawn()

The basic pattern for using Modal as a job queue involves three key steps:

1. Defining and deploying the job processing function using `modal deploy`.
2. Submitting a job using
   [`modal.Function.spawn()`](https://modal.com/docs/reference/modal.Function#spawn)
3. Polling for the job's result using
   [`modal.FunctionCall.get()`](https://modal.com/docs/reference/modal.FunctionCall#get)

Here's a simple example that you can run with `modal run my_job_queue.py`:

```python
# my_job_queue.py
import modal

app = modal.App("my-job-queue")

@app.function()
def process_job(data):
    # Perform the job processing here
    return {"result": data}

def submit_job(data):
    # Since the `process_job` function is deployed, need to first look it up
    process_job = modal.Function.from_name("my-job-queue", "process_job")
    call = process_job.spawn(data)
    return call.object_id

def get_job_result(call_id):
    function_call = modal.FunctionCall.from_id(call_id)
    try:
        result = function_call.get(timeout=5)
    except modal.exception.OutputExpiredError:
        result = {"result": "expired"}
    except TimeoutError:
        result = {"result": "pending"}
    return result

@app.local_entrypoint()
def main():
    data = "my-data"

    # Submit the job to Modal
    call_id = submit_job(data)
    print(get_job_result(call_id))
```

In this example:

- `process_job` is the Modal function that performs the actual job processing.
  To deploy the `process_job` function on Modal, run
  `modal deploy my_job_queue.py`.
- `submit_job` submits a new job by first looking up the deployed `process_job`
  function, then calling `.spawn()` with the job data. It returns the unique ID
  of the spawned function call.
- `get_job_result` attempts to retrieve the result of a previously submitted job
  using [`FunctionCall.from_id()`](https://modal.com/docs/reference/modal.FunctionCall#from_id) and
  [`FunctionCall.get()`](https://modal.com/docs/reference/modal.FunctionCall#get).
  [`FunctionCall.get()`](https://modal.com/docs/reference/modal.FunctionCall#get) waits indefinitely
  by default. It takes an optional timeout argument that specifies the maximum
  number of seconds to wait, which can be set to 0 to poll for an output
  immediately. Here, if the job hasn't completed yet, we return a pending
  response.
- The results of a `.spawn()` are accessible via `FunctionCall.get()` for up to
  7 days after completion. After this period, we return an expired response.

[Document OCR Web App](https://modal.com/docs/examples/doc_ocr_webapp) is an example that uses
this pattern.

## Integration with web frameworks

You can easily integrate the job queue pattern with web frameworks like FastAPI.
Here's an example, assuming that you have already deployed `process_job` on
Modal with `modal deploy` as above. This example won't work if you haven't
deployed your app yet.

```python
# my_job_queue_endpoint.py
import modal

image = modal.Image.debian_slim().pip_install("fastapi[standard]")
app = modal.App("fastapi-modal", image=image)

@app.function()
@modal.asgi_app()
@modal.concurrent(max_inputs=20)
def fastapi_app():
    from fastapi import FastAPI

    web_app = FastAPI()

    @web_app.post("/submit")
    async def submit_job_endpoint(data):
        process_job = modal.Function.from_name("my-job-queue", "process_job")

        call = await process_job.spawn.aio(data)
        return {"call_id": call.object_id}

    @web_app.get("/result/{call_id}")
    async def get_job_result_endpoint(call_id: str):
        function_call = modal.FunctionCall.from_id(call_id)
        try:
            result = await function_call.get.aio(timeout=0)
        except modal.exception.OutputExpiredError:
            return fastapi.responses.JSONResponse(content="", status_code=404)
        except TimeoutError:
            return fastapi.responses.JSONResponse(content="", status_code=202)

        return result

    return web_app
```

In this example:

- The `/submit` endpoint accepts job data, submits a new job using
  `await process_job.spawn.aio()`, and returns the job's ID to the client.
- The `/result/{call_id}` endpoint allows the client to poll for the job's
  result using the job ID. If the job hasn't completed yet, it returns a 202
  status code to indicate that the job is still being processed. If the job
  has expired, it returns a 404 status code to indicate that the job is not found.

You can try this app by serving it with `modal serve`:

```shell
modal serve my_job_queue_endpoint.py
```

Then interact with its endpoints with `curl`:

```shell
# Make a POST request to your app endpoint with.
$ curl -X POST $YOUR_APP_ENDPOINT/submit?data=data
{"call_id":"fc-XXX"}

# Use the call_id value from above.
$ curl -X GET $YOUR_APP_ENDPOINT/result/fc-XXX
```

## Scaling and reliability

Modal automatically scales the job queue based on the workload, spinning up new
instances as needed to process jobs concurrently. It also provides built-in
reliability features like automatic retries and timeout handling.

You can customize the behavior of the job queue by configuring the
`@app.function()` decorator with options like
[`retries`](https://modal.com/docs/guide/retries#function-retries),
[`timeout`](https://modal.com/docs/guide/timeouts#timeouts), and
[`max_containers`](https://modal.com/docs/guide/scale#configuring-autoscaling-behavior).

#### Dynamic batching (beta)

# Dynamic batching (beta)

Modal's `@batched` feature allows you to accumulate requests
and process them in dynamically-sized batches, rather than one-by-one.

Batching increases throughput at a potential cost to latency.
Batched requests can share resources and reuse work, reducing the time and cost per request.
Batching is particularly useful for GPU-accelerated machine learning workloads,
as GPUs are designed to maximize throughput and are frequently bottlenecked on shareable resources,
like weights stored in memory.

Static batching can lead to unbounded latency, as the function waits for a fixed number of requests to arrive.
Modal's dynamic batching waits for the lesser of a fixed time _or_ a fixed number of requests before executing,
maximizing the throughput benefit of batching while minimizing the latency penalty.

## Enable dynamic batching with `@batched`

To enable dynamic batching, apply the
[`@modal.batched` decorator](https://modal.com/docs/reference/modal.batched) to the target
Python function. Then, wrap it in `@app.function()` and run it on Modal,
and the inputs will be accumulated and processed in batches.

Here's what that looks like:

```python
import modal

app = modal.App()

@app.function()
@modal.batched(max_batch_size=2, wait_ms=1000)
async def batch_add(xs: list[int], ys: list[int]) -> list[int]:
    return [x + y for x, y in zip(xs, ys)]
```

When you invoke a function decorated with `@batched`, you invoke it asynchronously on individual inputs.
Outputs are returned where they were invoked.

For instance, the code below invokes the decorated `batch_add` function above three times, but `batch_add`
only executes twice:

```python continuation
@app.local_entrypoint()
async def main():
    inputs = [(1, 300), (2, 200), (3, 100)]
    async for result in batch_add.starmap.aio(inputs):
        print(f"Sum: {result}")
        # Sum: 301
        # Sum: 202
        # Sum: 103
```

The first time it is executed with `xs` batched to `[1, 2]`
and `ys` batched to `[300, 200]`. After about a one second delay, it is executed with `xs`
batched to `[3]` and `ys` batched to `[100]`.
The result is an iterator that yields `301`, `202`, and `101`.

## Use `@batched` with functions that take and return lists

For a Python function to be compatible with `@modal.batched`, it must adhere to
the following rules:

- ** The inputs to the function must be lists. **
  In the example above, we pass `xs` and `ys`, which are both lists of `int`s.
- ** The function must return a list**. In the example above, the function returns
  a list of sums.
- ** The lengths of all the input lists and the output list must be the same. **
  In the example above, if `L == len(xs) == len(ys)`, then `L == len(batch_add(xs, ys))`.

## Modal `Cls` methods are compatible with dynamic batching

Methods on Modal [`Cls`](https://modal.com/docs/guide/lifecycle-functions)es also support dynamic batching.

```python
import modal

app = modal.App()

@app.cls()
class BatchedClass():
    @modal.batched(max_batch_size=2, wait_ms=1000)
    async def batch_add(self, xs: list[int], ys: list[int]) -> list[int]:
        return [x + y for x, y in zip(xs, ys)]
```

One additional rule applies to classes with Batched Methods:

- If a class has a Batched Method, it **cannot have other Batched Methods or [Methods](https://modal.com/docs/reference/modal.method#modalmethod)**.

## Configure the wait time and batch size of dynamic batches

The `@batched` decorator takes in two required configuration parameters:

- `max_batch_size` limits the number of inputs combined into a single batch.
- `wait_ms` limits the amount of time the Function waits for more inputs after
  the first input is received.

The first invocation of the Batched Function initiates a new batch, and subsequent
calls add requests to this ongoing batch. If `max_batch_size` is reached,
the batch immediately executes. If the `max_batch_size` is not met but `wait_ms`
has passed since the first request was added to the batch, the unfilled batch is
executed.

### Selecting a batch configuration

To optimize the batching configurations for your application, consider the following heuristics:

- Set `max_batch_size` to the largest value your function can handle, so you
  can amortize and parallelize as much work as possible.

- Set `wait_ms` to the difference between your targeted latency and the execution time. Most applications
  have a targeted latency, and this allows the latency of any request to stay
  within that limit.

## Serve web endpoints with dynamic batching

Here's a simple example of serving a Function that batches requests dynamically
with a [`@modal.fastapi_endpoint`](https://modal.com/docs/guide/webhooks). Run
[`modal serve`](https://modal.com/docs/reference/cli/serve), submit requests to the endpoint,
and the Function will batch your requests on the fly.

```python
import modal

app = modal.App(image=modal.Image.debian_slim().pip_install("fastapi"))

@app.function()
@modal.batched(max_batch_size=2, wait_ms=1000)
async def batch_add(xs: list[int], ys: list[int]) -> list[int]:
    return [x + y for x, y in zip(xs, ys)]

@app.function()
@modal.fastapi_endpoint(method="POST", docs=True)
async def add(body: dict[str, int]) -> dict[str, int]:
    result = await batch_add.remote.aio(body["x"], body["y"])
    return {"result": result}
```

Now, you can submit requests to the web endpoint and process them in batches. For instance, the three requests
in the following example, which might be requests from concurrent clients in a real deployment,
will be batched into two executions:

```python notest
import asyncio
import aiohttp

async def send_post_request(session, url, data):
    async with session.post(url, json=data) as response:
        return await response.json()

async def main():
    # Enter the URL of your web endpoint here
    url = "https://workspace--app-name-endpoint-name.modal.run"

    async with aiohttp.ClientSession() as session:
        # Submit three requests asynchronously
        tasks = [
            send_post_request(session, url, {"x": 1, "y": 300}),
            send_post_request(session, url, {"x": 2, "y": 200}),
            send_post_request(session, url, {"x": 3, "y": 100}),
        ]
        results = await asyncio.gather(*tasks)
        for result in results:
            print(f"Sum: {result['result']}")

asyncio.run(main())
```

#### Multi-node clusters (beta)

# Multi-node clusters (beta)

> 🚄 Multi-node clusters with RDMA are in **private beta.** Please contact us via the [Modal Slack](https://modal.com/slack) or support@modal.com to get access.

Modal supports running a training job across several coordinated containers. Each container can saturate the available GPU devices on its host (a.k.a node) and communicate with peer containers which do the same. By scaling a training job from a single GPU to 16 GPUs you can achieve nearly 16x improvements in training time.

### Cluster compute capability

Modal H100 clusters provide:

- A 50 Gbps [IPv6 private network](https://modal.com/docs/guide/private-networking) for orchestration, dataset downloading.
- A 3200 Gbps RDMA scale-out network ([RoCE](https://en.wikipedia.org/wiki/RDMA_over_Converged_Ethernet)).
- Up-to 64 H100 SXM devices.
- At least 1TB of RAM and 4TB of local NVMe SSD per node.
- Deep burn-in testing.
- Interopability with all Modal platform functionality (Volumes, Dicts, Tunnels, etc.).

The guide will walk you through how the Modal client library enables multi-node training and integrates with `torchrun`.

### @clustered

Unlike standard Modal serverless containers, containers in a multi-node training job must be able to:

1. Perform fast, direct network communication between each other.
2. Be scheduled together, all or nothing, at the same time.

The `@clustered` decorator enables this behavior.

```python notest
import modal
import modal.experimental

@app.function(
    gpu="H100:8",
    timeout=60 * 60 * 24,
    retries=modal.Retries(initial_delay=0.0, max_retries=10),
)
@modal.experimental.clustered(size=4)
def train_model():
    cluster_info = modal.experimental.get_cluster_info()

    container_rank = cluster_info.rank
    world_size = len(cluster_info.container_ips)
    main_addr = cluster_info.container_ips[0]
    is_main = "(main)" if container_rank == 0 else ""

    print(f"{container_rank=} {is_main} {world_size=} {main_addr=}")
    ...
```

Applying this decorator under `@app.function` modifies the Function so that remote calls to it are serviced by a multi-node container group. The above configuration creates a group of four containers each having 8 H100 GPU devices, for a total of 32 devices.

## Scheduling

A `modal.experimental.clustered` Function runs on multiple nodes in our cloud, but executes like a normal function call. For example, all nodes are scheduled together ([gang scheduling](https://en.wikipedia.org/wiki/Gang_scheduling)) so that your code runs on all of the requested hardware or not at all.

Traditionally this kind of cluster and scheduling management would be handled by SLURM, Kubernetes, or manually. But with Modal it’s all provided serverlessly with just an application of the decorator!

### Rank & input broadcast

![diagram](https://modal-cdn.com/cdnbot/multinodepmgnla70_4b57a155.webp)

You may notice above that a single `.remote` Function call created three input executions but returned only one output. This is how input-output is structured for multi-node training jobs on Modal. The Function call’s arguments are replicated to each container, but only the rank zero container’s is returned to the caller.

A container’s rank is a key concept in multi-node training jobs. Rank zero is the ‘leader’ rank and typically coordinates the job. Rank zero is also known as the “main” container. Rank zero’s output will always be the output of a multi-node training run.

## Networking

Function containers cannot normally make direct network connections to other Function containers, but this is a requirement for multi-node training communication. So, along with gang scheduling, the `@clustered` decorator enables Modal’s workspace-private inter-container networking called [i6pn](https://www.notion.so/Multi-node-docs-1281e7f16949806f966adedfe8b2cb74?pvs=21).

The [cluster networking guide](https://modal.com/docs/guide/private-networking) goes into more detail on i6pn, but the upshot is that each container in the cluster is made aware of the network address of all the other containers in the cluster, enabling them to communicate with each other quickly via [TCP](https://pytorch.org/docs/stable/elastic/rendezvous.html).

### RDMA (Infiniband)

H100 clusters are equipped with Infiniband providing up-to 3,200Gbps scale-out bandwidth for inter-node communication.
RDMA scale-out networking is enabled with the `rdma` parameter of `modal.experimental.clustered.`

```python notest
@modal.experimental.clustered(size=2, rdma=True)
def train():
    ...
```

To run a simple Infiniband RDMA performance test see the [`modal-examples` repository example](https://github.com/modal-labs/multinode-training-guide/tree/main/benchmark).

## Cluster Info

`modal.experimental.get_cluster_info()` exposes the following information about the cluster:

- `rank: int` is the current container's order within the cluster, starting from `0`, the leader.
- `container_ips: list[str]` contains the ipv6 addresses of each container in the cluster, sorted by rank.
- `container_v4_ips: list[str]` contains the ipv4 addresses of each container in the cluster, sorted by rank.

## Fault Tolerance

For a clustered Function, failures in inputs and containers are handled differently.

If an input fails on any container, this failure **is not propagated** to other containers in the cluster. Containers are responsible for detecting and responding to input failures on other containers.

Only rank 0’s output matters: if an input fails on the leader container (rank 0), the input is marked as failed, even if the input succeeds on another container. Similarly, if an input succeeds on the leader container but fails on another container, the input will still be marked as successful.

If a container in the cluster is preempted, Modal will terminate all remaining containers in the cluster, and retry the input.

### Input Synchronization

_**Important:**_ synchronization is not relevant for single training runs, and applies mostly to inference use-cases.

Modal does not synchronize input execution across containers. Containers are responsible for ensuring that they do not process inputs faster than other containers in their cluster.

In particular, it is important that the leader container (rank 0) only starts processing the next input after all other containers have finished processing the current input.

## Examples

To get hands-on with multi-node training you can jump into the [`multinode-training-guide` repository](https://github.com/modal-labs/multinode-training-guide) or [`modal-examples` repository](https://github.com/modal-labs/modal-examples/tree/main/12_datasets) and `modal run` something!

- [Simple ‘hello world’ 4 X 1 H100 torch cluster example](https://github.com/modal-labs/modal-examples/blob/main/14_clusters/simple_torch_cluster.py)
- [Infiniband RDMA performance test](https://github.com/modal-labs/multinode-training-guide/tree/main/benchmark)
- [Use 2 x 8 H100s to train a ResNet50 model on the ImageNet dataset](https://github.com/modal-labs/multinode-training-guide/tree/main/resnet50)
- [Speedrun GPT-2 training with modded-nanogpt](https://github.com/modal-labs/multinode-training-guide/tree/main/nanoGPT)
<!-- - Use 2 x 8 H100s to run multi-node _inference_ on LLaMA 3.1 405B in 16bit precision. **[TODO]** -->

### Torchrun Example

```python
import modal
import modal.experimental

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch~=2.5.1", "numpy~=2.2.1")
    .add_local_dir(
        "training", remote_path="/root/training"
    )
)
app = modal.App("example-simple-torch-cluster", image=image)

n_nodes = 4

@app.function(
    gpu=f"H100:8",
    timeout=60 * 60 * 24,
)
@modal.experimental.clustered(size=n_nodes)
def launch_torchrun():
    # import the 'torchrun' interface directly.
    from torch.distributed.run import parse_args, run

    cluster_info = modal.experimental.get_cluster_info()

    run(
        parse_args(
            [
                f"--nnodes={n_nodes}",
                f"--node_rank={cluster_info.rank}",
                f"--master_addr={cluster_info.container_ips[0]}",
                f"--nproc-per-node=8",
                "--master_port=1234",
                "training/train.py",
            ]
        )
    )
```

### Scheduling and cron jobs

# Scheduling remote cron jobs

A common requirement is to perform some task at a given time every day or week
automatically. Modal facilitates this through function schedules.

## Basic scheduling

Let's say we have a Python module `heavy.py` with a function,
`perform_heavy_computation()`.

```python
# heavy.py
def perform_heavy_computation():
    ...

if __name__ == "__main__":
    perform_heavy_computation()
```

To schedule this function to run once per day, we create a Modal App and attach
our function to it with the `@app.function` decorator and a schedule parameter:

```python
# heavy.py
import modal

app = modal.App()

@app.function(schedule=modal.Period(days=1))
def perform_heavy_computation():
    ...
```

To activate the schedule, deploy your app, either through the CLI:

```shell
modal deploy --name daily_heavy heavy.py
```

Or programmatically:

```python
if __name__ == "__main__":
   app.deploy()
```

Now the function will run every day, at the time of the initial deployment,
without any further interaction on your part.

When you make changes to your function, just rerun the deploy command to
overwrite the old deployment.

Note that when you redeploy your function, `modal.Period` resets, and the
schedule will run X hours after this most recent deployment.

If you want to run your function at a regular schedule not disturbed by deploys,
`modal.Cron` (see below) is a better option.

## Monitoring your scheduled runs

To see past execution logs for the scheduled function, go to the
[Apps](https://modal.com/apps) section on the Modal web site.

Schedules currently cannot be paused. Instead the schedule should be removed and
the app redeployed. Schedules can be started manually on the app's dashboard
page, using the "run now" button.

## Schedule types

There are two kinds of base schedule values -
[`modal.Period`](https://modal.com/docs/reference/modal.Period) and
[`modal.Cron`](https://modal.com/docs/reference/modal.Cron).

[`modal.Period`](https://modal.com/docs/reference/modal.Period) lets you specify an interval
between function calls, e.g. `Period(days=1)` or `Period(hours=5)`:

```python
# runs once every 5 hours
@app.function(schedule=modal.Period(hours=5))
def perform_heavy_computation():
    ...
```

[`modal.Cron`](https://modal.com/docs/reference/modal.Cron) gives you finer control using
[cron](https://en.wikipedia.org/wiki/Cron) syntax:

```python
# runs at 8 am (UTC) every Monday
@app.function(schedule=modal.Cron("0 8 * * 1"))
def perform_heavy_computation():
    ...

# runs daily at 6 am (New York time)
@app.function(schedule=modal.Cron("0 6 * * *", timezone="America/New_York"))
def send_morning_report():
    ...
```

For more details, see the API reference for
[Period](https://modal.com/docs/reference/modal.Period), [Cron](https://modal.com/docs/reference/modal.Cron) and
[Function](https://modal.com/docs/reference/modal.Function)

### Deployment

#### Apps, Functions, and entrypoints

# Apps, Functions, and entrypoints

An [`App`](https://modal.com/docs/reference/modal.App) represents an application running on Modal. It groups one or more Functions for atomic deployment and acts as a shared namespace. All Functions and Clses are associated with an
App.

A [`Function`](https://modal.com/docs/reference/modal.Function) acts as an independent unit once it is deployed, and [scales up and down](https://modal.com/docs/guide/scale) independently from other Functions. If there are no live inputs to the Function then by default, no containers will run and your account will not be charged for compute resources, even if the App it belongs to is deployed.

An App can be ephemeral or deployed. You can view a list of all currently running Apps on the [`apps`](https://modal.com/apps) page.

The code for a Modal App defining two separate Functions might look something like this:

```python

import modal

app = modal.App(name="my-modal-app")

@app.function()
def f():
    print("Hello world!")

@app.function()
def g():
    print("Goodbye world!")

```

## Ephemeral Apps

An ephemeral App is created when you use the
[`modal run`](https://modal.com/docs/reference/cli/run) CLI command, or the
[`app.run`](https://modal.com/docs/reference/modal.App#run) method. This creates a temporary
App that only exists for the duration of your script.

Ephemeral Apps are stopped automatically when the calling program exits, or when
the server detects that the client is no longer connected.
You can use
[`--detach`](https://modal.com/docs/reference/cli/run) in order to keep an ephemeral App running even
after the client exits.

By using `app.run` you can run your Modal apps from within your Python scripts:

```python
def main():
    ...
    with app.run():
        some_modal_function.remote()
```

By default, running your app in this way won't propagate Modal logs and progress bar messages. To enable output, use the [`modal.enable_output`](https://modal.com/docs/reference/modal.enable_output) context manager:

```python
def main():
    ...
    with modal.enable_output():
        with app.run():
            some_modal_function.remote()
```

## Deployed Apps

A deployed App is created using the [`modal deploy`](https://modal.com/docs/reference/cli/deploy)
CLI command. The App is persisted indefinitely until you delete it via the
[web UI](https://modal.com/apps). Functions in a deployed App that have an attached
[schedule](https://modal.com/docs/guide/cron) will be run on a schedule. Otherwise, you can
invoke them manually using
[web endpoints or Python](https://modal.com/docs/guide/trigger-deployed-functions).

Deployed Apps are named via the [`App`](https://modal.com/docs/reference/modal.App#modalapp)
constructor. Re-deploying an existing `App` (based on the name) will update it
in place.

## Entrypoints for ephemeral Apps

The code that runs first when you `modal run` an App is called the "entrypoint".

You can register a local entrypoint using the
[`@app.local_entrypoint()`](https://modal.com/docs/reference/modal.App#local_entrypoint)
decorator. You can also use a regular Modal function as an entrypoint, in which
case only the code in global scope is executed locally.

### Argument parsing

If your entrypoint function takes arguments with primitive types, `modal run`
automatically parses them as CLI options. For example, the following function
can be called with `modal run script.py --foo 1 --bar "hello"`:

```python
# script.py

@app.local_entrypoint()
def main(foo: int, bar: str):
    some_modal_function.remote(foo, bar)
```

If you wish to use your own argument parsing library, such as `argparse`, you can instead accept a variable-length argument list for your entrypoint or your function. In this case, Modal skips CLI parsing and forwards CLI arguments as a tuple of strings. For example, the following function can be invoked with `modal run my_file.py --foo=42 --bar="baz"`:

```python
import argparse

@app.function()
def train(*arglist):
    parser = argparse.ArgumentParser()
    parser.add_argument("--foo", type=int)
    parser.add_argument("--bar", type=str)
    args = parser.parse_args(args = arglist)
```

### Manually specifying an entrypoint

If there is only one `local_entrypoint` registered,
[`modal run script.py`](https://modal.com/docs/reference/cli/run) will automatically use it. If
you have no entrypoint specified, and just one decorated Modal function, that
will be used as a remote entrypoint instead. Otherwise, you can direct
`modal run` to use a specific entrypoint.

For example, if you have a function decorated with
[`@app.function()`](https://modal.com/docs/reference/modal.App#function) in your file:

```python
# script.py

@app.function()
def f():
    print("Hello world!")

@app.function()
def g():
    print("Goodbye world!")

@app.local_entrypoint()
def main():
    f.remote()
```

Running [`modal run script.py`](https://modal.com/docs/reference/cli/run) will execute the `main`
function locally, which would call the `f` function remotely. However you can
instead run `modal run script.py::app.f` or `modal run script.py::app.g` to
execute `f` or `g` directly.

## Apps were once Stubs

The `modal.App` class in the client was previously called `modal.Stub`. The
old name was kept as an alias for some time, but from Modal 1.0.0 onwards,
using `modal.Stub` will result in an error.

#### Managing deployments

# Managing deployments

Once you've finished using `modal run` or `modal serve` to iterate on your Modal
code, it's time to deploy. A Modal deployment creates and then persists an
application and its objects, providing the following benefits:

- Repeated application function executions will be grouped under the deployment,
  aiding observability and usage tracking. Programmatically triggering lots of
  ephemeral App runs can clutter your web and CLI interfaces.
- Function calls are much faster because deployed functions are persistent and
  reused, not created on-demand by calls. Learn how to trigger deployed
  functions in
  [Invoking deployed functions](https://modal.com/docs/guide/trigger-deployed-functions).
- [Scheduled functions](https://modal.com/docs/guide/cron) will continue scheduling separate from
  any local iteration you do, and will notify you on failure.
- [Web endpoints](https://modal.com/docs/guide/webhooks) keep running when you close your laptop,
  and their URL address matches the deployment name.

## Creating deployments

Deployments are created using the
[`modal deploy` command](https://modal.com/docs/reference/cli/app#modal-app-list).

```
 % modal deploy -m whisper_pod_transcriber.main
✓ Initialized. View app page at https://modal.com/apps/ap-PYc2Tb7JrkskFUI8U5w0KG.
✓ Created objects.
├── 🔨 Created populate_podcast_metadata.
├── 🔨 Mounted /home/ubuntu/whisper_pod_transcriber at /root/whisper_pod_transcriber
├── 🔨 Created fastapi_app => https://modal-labs-whisper-pod-transcriber-fastapi-app.modal.run
├── 🔨 Mounted /home/ubuntu/whisper_pod_transcriber/whisper_frontend/dist at /assets
├── 🔨 Created search_podcast.
├── 🔨 Created refresh_index.
├── 🔨 Created transcribe_segment.
├── 🔨 Created transcribe_episode..
└── 🔨 Created fetch_episodes.
✓ App deployed! 🎉

View Deployment: https://modal.com/apps/modal-labs/whisper-pod-transcriber
```

Running this command on an existing deployment will redeploy the App,
incrementing its version. For detail on how live deployed apps transition
between versions, see the [Updating deployments](#updating-deployments) section.

Deployments can also be created programmatically using Modal's
[Python API](https://modal.com/docs/reference/modal.App#deploy).

## Viewing deployments

Deployments can be viewed either on the [apps](https://modal.com/apps) web page or by using the
[`modal app list` command](https://modal.com/docs/reference/cli/app#modal-app-list).

## Updating deployments

A deployment can deploy a new App or redeploy a new version of an existing
deployed App. It's useful to understand how Modal handles the transition between
versions when an App is redeployed. In general, Modal aims to support
zero-downtime deployments by gradually transitioning traffic to the new version.

If the deployment involves building new versions of the Images used by the App,
the build process will need to complete succcessfully. The existing version of
the App will continue to handle requests during this time. Errors during the
build will abort the deployment with no change to the status of the App.

After the build completes, Modal will start to bring up new containers running
the latest version of the App. The existing containers will continue handling
requests (using the previous version of the App) until the new containers have
completed their cold start.

Once the new containers are ready, old containers will stop accepting new
requests. However, the old containers will continue running any requests they
had previously accepted. The old containers will not terminate until they have
finished processing all ongoing requests.

Any warm pool containers will also be cycled during a deployment, as the
previous version's warm pool are now outdated.

## Deployment rollbacks

To quickly reset an App back to a previous version, you can perform a deployment
_rollback_. Rollbacks can be triggered from either the App dashboard or the CLI.
Rollback deployments look like new deployments: they increment the version number
and are attributed to the user who triggered the rollback. But the App's functions
and metadata will be reset to their previous state independently of your current
App codebase.

Note that deployment rollbacks are supported only on the Team and Enterprise plans.

## Stopping deployments

Deployed apps can be stopped in the web UI by clicking the red "Stop app" button on
the App's "Overview" page, or alternatively from the command line using the
[`modal app stop` command](https://modal.com/docs/reference/cli/app#modal-app-stop).

Stopping an App is a destructive action. Apps cannot be restarted from this state;
a new App will need to be deployed from the same source files. Objects associated
with stopped deployments will eventually be garbage collected.

#### Invoking deployed functions

# Invoking deployed functions

Modal lets you take a function created by a
[deployment](https://modal.com/docs/guide/managing-deployments) and call it from other contexts.

There are two ways of invoking deployed functions. If the invoking client is
running Python, then the same
[Modal client library](https://pypi.org/project/modal/) used to write Modal code
can be used. HTTPS is used if the invoking client is not running Python and
therefore cannot import the Modal client library.

## Invoking with Python

Some use cases for Python invocation include:

- An existing Python web server (eg. Django, Flask) wants to invoke Modal
  functions.
- You have split your product or system into multiple Modal applications that
  deploy independently and call each other.

### Function lookup and invocation basics

Let's say you have a script `my_shared_app.py` and this script defines a Modal
app with a function that computes the square of a number:

```python
import modal

app = modal.App("my-shared-app")

@app.function()
def square(x: int):
    return x ** 2
```

You can deploy this app to create a persistent deployment:

```
% modal deploy shared_app.py
✓ Initialized.
✓ Created objects.
├── 🔨 Created square.
├── 🔨 Mounted /Users/erikbern/modal/shared_app.py.
✓ App deployed! 🎉

View Deployment: https://modal.com/apps/erikbern/my-shared-app
```

Let's try to run this function from a different context. For instance, let's
fire up the Python interactive interpreter:

```bash
% python
Python 3.9.5 (default, May  4 2021, 03:29:30)
[Clang 12.0.0 (clang-1200.0.32.27)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import modal
>>> f = modal.Function.from_name("my-shared-app", "square")
>>> f.remote(42)
1764
>>>
```

This works exactly the same as a regular modal `Function` object. For example,
you can `.map()` over functions invoked this way too:

```bash
>>> f = modal.Function.from_name("my-shared-app", "square")
>>> f.map([1, 2, 3, 4, 5])
[1, 4, 9, 16, 25]
```

#### Authentication

The Modal Python SDK will read the token from `~/.modal.toml` which typically is
created using `modal token new`.

Another method of providing the credentials is to set the environment variables
`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. If you want to call a Modal function
from a context such as a web server, you can expose these environment variables
to the process.

#### Lookup of lifecycle functions

[Lifecycle functions](https://modal.com/docs/guide/lifecycle-functions) are defined on classes,
which you can look up in a different way. Consider this code:

```python
import modal

app = modal.App("my-shared-app")

@app.cls()
class MyLifecycleClass:
    @modal.enter()
    def enter(self):
        self.var = "hello world"

    @modal.method()
    def foo(self):
        return self.var
```

Let's say you deploy this app. You can then call the function by doing this:

```bash
>>> cls = modal.Cls.from_name("my-shared-app", "MyLifecycleClass")
>>> obj = cls()  # You can pass any constructor arguments here
>>> obj.foo.remote()
'hello world'
```

### Asynchronous invocation

In certain contexts, a Modal client will need to trigger Modal functions without
waiting on the result. This is done by spawning functions and receiving a
[`FunctionCall`](https://modal.com/docs/reference/modal.FunctionCall) as a
handle to the triggered execution.

The following is an example of a Flask web server (running outside Modal) which
accepts model training jobs to be executed within Modal. Instead of the HTTP
POST request waiting on a training job to complete, which would be infeasible,
the relevant Modal function is spawned and the
[`FunctionCall`](https://modal.com/docs/reference/modal.FunctionCall)
object is stored for later polling of execution status.

```python
from uuid import uuid4
from flask import Flask, jsonify, request

app = Flask(__name__)
pending_jobs = {}

...

@app.route("/jobs", methods = ["POST"])
def create_job():
    predict_fn = modal.Function.from_name("example", "train_model")
    job_id = str(uuid4())
    function_call = predict_fn.spawn(
        job_id=job_id,
        params=request.json,
    )
    pending_jobs[job_id] = function_call
    return {
        "job_id": job_id,
        "status": "pending",
    }
```

### Importing a Modal function between Modal apps

You can also import one function defined in an app from another app:

```python
import modal

app = modal.App("another-app")

square = modal.Function.from_name("my-shared-app", "square")

@app.function()
def cube(x):
    return x * square.remote(x)

@app.local_entrypoint()
def main():
    assert cube.remote(42) == 74088
```

### Comparison with HTTPS

Compared with HTTPS invocation, Python invocation has the following benefits:

- Avoids the need to create web endpoint functions.
- Avoids handling serialization of request and response data between Modal and
  your client.
- Uses the Modal client library's built-in authentication.
  - Web endpoints are public to the entire internet, whereas function `lookup`
    only exposes your code to you (and your org).
- You can work with shared Modal functions as if they are normal Python
  functions, which might be more convenient.

## Invoking with HTTPS

Any non-Python application client can interact with deployed Modal applications
via [web endpoint functions](https://modal.com/docs/guide/webhooks).

Anything able to make HTTPS requests can trigger a Modal web endpoint function.
Note that all deployed web endpoint functions have
[a stable HTTPS URL](https://modal.com/docs/guide/webhook-urls).

Some use cases for HTTPS invocation include:

- Calling Modal functions from a web browser client running Javascript
- Calling Modal functions from non-Python backend services (Java, Go, Ruby,
  NodeJS, etc)
- Calling Modal functions using UNIX tools (`curl`, `wget`)

However, if the client of your Modal deployment is running Python, it's better
to use the [Modal client library](https://pypi.org/project/modal/) to invoke
your Modal code.

For more detail on setting up functions for invocation over HTTP see the
[web endpoints guide](https://modal.com/docs/guide/webhooks).

#### Continuous deployment

# Continuous deployment

It's a common pattern to auto-deploy your Modal App as part of a CI/CD pipeline.
To get you started, below is a guide to doing continuous deployment of a Modal
App in GitHub.

## GitHub Actions

Here's a sample GitHub Actions workflow that deploys your App on every push to
the `main` branch.

This requires you to create a [Modal token](https://modal.com/settings/tokens) and add it as a
[secret for your Github Actions workflow](https://github.com/Azure/actions-workflow-samples/blob/master/assets/create-secrets-for-GitHub-workflows.md).

After setting up secrets, create a new workflow file in your repository at
`.github/workflows/ci-cd.yml` with the following contents:

```yaml
name: CI/CD

on:
  push:
    branches:
      - main

jobs:
  deploy:
    name: Deploy
    runs-on: ubuntu-latest
    env:
      MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}
      MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Install Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install Modal
        run: |
          python -m pip install --upgrade pip
          pip install modal

      - name: Deploy job
        run: |
          modal deploy -m my_package.my_file
```

Be sure to replace `my_package.my_file` with your actual entrypoint.

If you use multiple Modal [Environments](https://modal.com/docs/guide/environments), you can
additionally specify the target environment in the YAML using
`MODAL_ENVIRONMENT=xyz`.

#### Running untrusted code in Functions

# Running untrusted code in Functions

Modal provides two primitives for running untrusted code: Restricted Functions and [Sandboxes](https://modal.com/docs/guide/sandbox). While both can be used for running untrusted code, they serve different purposes: Sandboxes provide a container-like interface while Restricted Functions provide an interface similar to a traditional Function.

Restricted Functions are useful for executing:

- Code generated by language models (LLMs)
- User-submitted code in interactive environments
- Third-party plugins or extensions

## Using `restrict_modal_access`

To restrict a Function's access to Modal resources, set `restrict_modal_access=True` on the Function definition:

```python
import modal

app = modal.App()

@app.function(restrict_modal_access=True)
def run_untrusted_code(code_input: str):
    # This function cannot access Modal resources
    return eval(code_input)
```

When `restrict_modal_access` is enabled:

- The Function cannot access Modal resources (Queues, Dicts, etc.)
- The Function cannot call other Functions
- The Function cannot access Modal's internal APIs

## Comparison with Sandboxes

While both `restrict_modal_access` and [Sandboxes](https://modal.com/docs/guide/sandbox) can be used for running untrusted code, they serve different purposes:

| Feature   | Restricted Function            | Sandbox                                        |
| --------- | ------------------------------ | ---------------------------------------------- |
| State     | Stateless                      | Stateful                                       |
| Interface | Function-like                  | Container-like                                 |
| Setup     | Simple decorator               | Requires explicit creation/termination         |
| Use case  | Quick, isolated code execution | Interactive development, long-running sessions |

## Best Practices

When running untrusted code, consider these additional security measures:

1. Use `max_inputs=1` to ensure each container only handles one request. Containers that get reused could cause information leakage between users.

```python
@app.function(restrict_modal_access=True, max_inputs=1)
def isolated_function(input_data):
    # Each input gets a fresh container
    return process(input_data)
```

2. Set appropriate timeouts to prevent long-running operations:

```python
@app.function(
    restrict_modal_access=True,
    timeout=30,  # 30 second timeout
    max_inputs=1
)
def time_limited_function(input_data):
    return process(input_data)
```

3. Consider using `block_network=True` to prevent the container from making outbound network requests:

```python
@app.function(
    restrict_modal_access=True,
    block_network=True,
    max_inputs=1
)
def network_isolated_function(input_data):
    return process(input_data)
```

4. Minimize the App source that's included in the container

A restricted Modal Function will have read access to its source files in the
container, so you'll want to avoid including anything that would be harmful
if exfiltrated by the untrusted process.

If deploying an App from within a [larger package](https://modal.com/docs/guide/project-structure),
the entire package source may be automatically included by default. A best
practice would be to make the untrusted Function part of a standalone App that
includes the minimum necessary files to run:

```python
restricted_app = modal.App("restricted-app", include_source=False)

image = (
    modal.Image.debian_slim()
    .add_local_file("restricted_executor.py", "/root/restricted_executor.py")
)

@restricted_app.function(
    restrict_modal_access=True,
    block_network=True,
    max_inputs=1,
)
def isolated_function(input_data):
    return process(input_data)
```

## Example: Running LLM-generated Code

Below is a complete example of running code generated by a language model:

```python
import modal

app = modal.App("restricted-access-example")

@app.function(restrict_modal_access=True, max_inputs=1, timeout=30, block_network=True)
def run_llm_code(generated_code: str):
    try:
        # Create a restricted environment
        execution_scope = {}

        # Execute the generated code
        exec(generated_code, execution_scope)

        # Return the result if it exists
        return execution_scope.get("result", None)
    except Exception as e:
        return f"Error executing code: {str(e)}"

@app.local_entrypoint()
def main():
    # Example LLM-generated code
    code = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

result = calculate_fibonacci(10)
    """

    result = run_llm_code.remote(code)
    print(f"Result: {result}")

```

This example locks down the container to ensure that the code is safe to execute by:

- Restricting Modal access
- Using a fresh container for each execution
- Setting a timeout
- Blocking network access
- Catching and handling potential errors

## Error Handling

When a restricted Function attempts to access Modal resources, it will raise an `AuthError`:

```python
@app.function(restrict_modal_access=True)
def restricted_function(q: modal.Queue):
    try:
        # This will fail because the Function is restricted
        return q.get()
    except modal.exception.AuthError as e:
        return f"Access denied: {e}"
```

The error message will indicate that the operation is not permitted due to restricted Modal access.

### Secrets and environment variables

#### Secrets

# Secrets

Securely provide credentials and other sensitive information to your Modal Functions with Secrets.

You can create and edit Secrets via
the [dashboard](https://modal.com/secrets),
the command line interface ([`modal secret`](https://modal.com/docs/reference/cli/secret)), and
programmatically from Python code ([`modal.Secret`](https://modal.com/docs/reference/modal.Secret)).

To inject Secrets into the container running your Function, add the
`secrets=[...]` argument to your `app.function` or `app.cls` decoration.

## Deploy Secrets from the Modal Dashboard

The most common way to create a Modal Secret is to use the
[Secrets panel of the Modal dashboard](https://modal.com/secrets),
which also shows any existing Secrets.

When you create a new Secret, you'll be prompted with a number of templates to help you get started.
These templates demonstrate standard formats for credentials for everything from Postgres and MongoDB
to Weights & Biases and Hugging Face.

## Use Secrets in your Modal Apps

You can then use your Secret by constructing it `from_name` when defining a Modal App
and then accessing its contents as environment variables.
For example, if you have a Secret called `secret-keys` containing the key
`MY_PASSWORD`:

```python
@app.function(secrets=[modal.Secret.from_name("secret-keys")])
def some_function():
    import os

    secret_key = os.environ["MY_PASSWORD"]
    ...
```

Each Secret can contain multiple keys and values but you can also inject
multiple Secrets, allowing you to separate Secrets into smaller reusable units:

```python
@app.function(secrets=[
    modal.Secret.from_name("my-secret-name"),
    modal.Secret.from_name("other-secret"),
])
def other_function():
    ...
```

The Secrets are applied in order, so key-values from later `modal.Secret`
objects in the list will overwrite earlier key-values in the case of a clash.
For example, if both `modal.Secret` objects above contained the key `FOO`, then
the value from `"other-secret"` would always be present in `os.environ["FOO"]`.

## Create Secrets programmatically

In addition to defining Secrets on the web dashboard, you can
programmatically create a Secret directly in your script and send it along to
your Function using `Secret.from_dict(...)`. This can be useful if you want to
send Secrets from your local development machine to the remote Modal App.

```python
import os

if modal.is_local():
    local_secret = modal.Secret.from_dict({"FOO": os.environ["LOCAL_FOO"]})
else:
    local_secret = modal.Secret.from_dict({})

@app.function(secrets=[local_secret])
def some_function():
    import os

    print(os.environ["FOO"])
```

If you have [`python-dotenv`](https://pypi.org/project/python-dotenv/) installed,
you can also use `Secret.from_dotenv()` to create a Secret from the variables in a `.env`
file

```python
@app.function(secrets=[modal.Secret.from_dotenv()])
def some_other_function():
    print(os.environ["USERNAME"])
```

## Interact with Secrets from the command line

You can create, list, and delete your Modal Secrets with the `modal secret` command line interface.

View your Secrets and their timestamps with

```bash
modal secret list
```

Create a new Secret by passing `{KEY}={VALUE}` pairs to `modal secret create`:

```bash
modal secret create database-secret PGHOST=uri PGPORT=5432 PGUSER=admin PGPASSWORD=hunter2
```

or using environment variables (assuming below that the `PGPASSWORD` environment variable is set
e.g. by your CI system):

```bash
modal secret create database-secret PGHOST=uri PGPORT=5432 PGUSER=admin PGPASSWORD="$PGPASSWORD"
```

Remove Secrets by passing their name to `modal secret delete`:

```bash
modal secret delete database-secret
```

#### Environment variables

# Environment variables

The Modal runtime sets several environment variables during initialization. The
keys for these environment variables are reserved and cannot be overridden by
your Function or Sandbox configuration.

These variables provide information about the containers's runtime
environment.

## Container runtime environment variables

The following variables are present in every Modal container:

- **`MODAL_CLOUD_PROVIDER`** — Modal executes containers across a number of cloud
  providers ([AWS](https://aws.amazon.com/), [GCP](https://cloud.google.com/),
  [OCI](https://www.oracle.com/cloud/)). This variable specifies which cloud
  provider the Modal container is running within.
- **`MODAL_IMAGE_ID`** — The ID of the
  [`modal.Image`](https://modal.com/docs/reference/modal.Image) used by the Modal container.
- **`MODAL_REGION`** — This will correspond to a geographic area identifier from
  the cloud provider associated with the Modal container (see above). For AWS, the
  identifier is a "region". For GCP it is a "zone", and for OCI it is an
  "availability domain". Example values are `us-east-1` (AWS), `us-central1`
  (GCP), `us-ashburn-1` (OCI). See the [full list here](https://modal.com/docs/guide/region-selection#region-options).
- **`MODAL_TASK_ID`** — The ID of the container running the Modal Function or Sandbox.

## Function runtime environment variables

The following variables are present in containers running Modal Functions:

- **`MODAL_ENVIRONMENT`** — The name of the
  [Modal Environment](https://modal.com/docs/guide/environments) the container is running within.
- **`MODAL_IS_REMOTE`** - Set to '1' to indicate that Modal Function code is running in
  a remote container.
- **`MODAL_IDENTITY_TOKEN`** — An [OIDC token](https://modal.com/docs/guide/oidc-integration)
  encoding the identity of the Modal Function.

## Sandbox environment variables

The following variables are present within [`modal.Sandbox`](https://modal.com/docs/reference/modal.Sandbox) instances.

- **`MODAL_SANDBOX_ID`** — The ID of the Sandbox.

## Container image environment variables

The container image layers used by a `modal.Image` may set
environment variables. These variables will be present within your container's runtime
environment. For example, the
[`debian_slim`](https://modal.com/docs/reference/modal.Image#debian_slim) image sets the
`GPG_KEY` variable.

To override image variables or set new ones, use the
[`.env`](https://modal.com/docs/reference/modal.Image#env) method provided by
`modal.Image`.

### Web endpoints

#### Web endpoints

# Web endpoints

This guide explains how to set up web endpoints with Modal.

All deployed Modal Functions can be [invoked from any other Python application](https://modal.com/docs/guide/trigger-deployed-functions)
using the Modal client library. We additionally provide multiple ways to expose
your Functions over the web for non-Python clients.

You can [turn any Python function into a web endpoint](#simple-endpoints) with a single line
of code, you can [serve a full app](#serving-asgi-and-wsgi-apps) using
frameworks like FastAPI, Django, or Flask, or you can
[serve anything that speaks HTTP and listens on a port](#non-asgi-web-servers).

Below we walk through each method, assuming you're familiar with web applications outside of Modal.
For a detailed walkthrough of basic web endpoints on Modal aimed at developers new to web applications,
see [this tutorial](https://modal.com/docs/examples/basic_web).

## Simple endpoints

The easiest way to create a web endpoint from an existing Python function is to use the
[`@modal.fastapi_endpoint` decorator](https://modal.com/docs/reference/modal.fastapi_endpoint).

```python
image = modal.Image.debian_slim().pip_install("fastapi[standard]")

@app.function(image=image)
@modal.fastapi_endpoint()
def f():
    return "Hello world!"
```

This decorator wraps the Modal Function in a
[FastAPI application](#how-do-web-endpoints-run-in-the-cloud).

_Note: Prior to v0.73.82, this function was named `@modal.web_endpoint`_.

### Developing with `modal serve`

You can run this code as an ephemeral app, by running the command

```shell
modal serve server_script.py
```

Where `server_script.py` is the file name of your code. This will create an
ephemeral app for the duration of your script (until you hit Ctrl-C to stop it).
It creates a temporary URL that you can use like any other REST endpoint. This
URL is on the public internet.

The `modal serve` command will live-update an app when any of its supporting
files change.

Live updating is particularly useful when working with apps containing web
endpoints, as any changes made to web endpoint handlers will show up almost
immediately, without requiring a manual restart of the app.

### Deploying with `modal deploy`

You can also deploy your app and create a persistent web endpoint in the cloud
by running `modal deploy`:

### Passing arguments to an endpoint

When using `@modal.fastapi_endpoint`, you can add
[query parameters](https://fastapi.tiangolo.com/tutorial/query-params/) which
will be passed to your Function as arguments. For instance

```python
image = modal.Image.debian_slim().pip_install("fastapi[standard]")

@app.function(image=image)
@modal.fastapi_endpoint()
def square(x: int):
    return {"square": x**2}
```

If you hit this with a URL-encoded query string with the `x` parameter present,
the Function will receive the value as an argument:

```
$ curl https://modal-labs--web-endpoint-square-dev.modal.run?x=42
{"square":1764}
```

If you want to use a `POST` request, you can use the `method` argument to
`@modal.fastapi_endpoint` to set the HTTP verb. To accept any valid JSON object,
[use `dict` as your type annotation](https://fastapi.tiangolo.com/tutorial/body-nested-models/?h=dict#bodies-of-arbitrary-dicts)
and FastAPI will handle the rest.

```python
image = modal.Image.debian_slim().pip_install("fastapi[standard]")

@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def square(item: dict):
    return {"square": item['x']**2}
```

This now creates an endpoint that takes a JSON body:

```
$ curl -X POST -H 'Content-Type: application/json' --data-binary '{"x": 42}' https://modal-labs--web-endpoint-square-dev.modal.run
{"square":1764}
```

This is often the easiest way to get started, but note that FastAPI recommends
that you use
[typed Pydantic models](https://fastapi.tiangolo.com/tutorial/body/) in order to
get automatic validation and documentation. FastAPI also lets you pass data to
web endpoints in other ways, for instance as
[form data](https://fastapi.tiangolo.com/tutorial/request-forms/) and
[file uploads](https://fastapi.tiangolo.com/tutorial/request-files/).

## How do web endpoints run in the cloud?

Note that web endpoints, like everything else on Modal, only run when they need
to. When you hit the web endpoint the first time, it will boot up the container,
which might take a few seconds. Modal keeps the container alive for a short
period in case there are subsequent requests. If there are a lot of requests,
Modal might create more containers running in parallel.

For the shortcut `@modal.fastapi_endpoint` decorator, Modal wraps your function in a
[FastAPI](https://fastapi.tiangolo.com/) application. This means that the
[Image](https://modal.com/docs/guide/images)
your Function uses must have FastAPI installed, and the Functions that you write
need to follow its request and response
[semantics](https://fastapi.tiangolo.com/tutorial). Web endpoint Functions can use
all of FastAPI's powerful features, such as Pydantic models for automatic validation,
typed query and path parameters, and response types.

Here's everything together, combining Modal's abilities to run functions in
user-defined containers with the expressivity of FastAPI:

```python
import modal
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

image = modal.Image.debian_slim().pip_install("fastapi[standard]", "boto3")
app = modal.App(image=image)

class Item(BaseModel):
    name: str
    qty: int = 42

@app.function()
@modal.fastapi_endpoint(method="POST")
def f(item: Item):
    import boto3
    # do things with boto3...
    return HTMLResponse(f"<html>Hello, {item.name}!</html>")
```

This endpoint definition would be called like so:

```bash
curl -d '{"name": "Erik", "qty": 10}' \
    -H "Content-Type: application/json" \
    -X POST https://ecorp--web-demo-f-dev.modal.run
```

Or in Python with the [`requests`](https://pypi.org/project/requests/) library:

```python
import requests

data = {"name": "Erik", "qty": 10}
requests.post("https://ecorp--web-demo-f-dev.modal.run", json=data, timeout=10.0)
```

## Serving ASGI and WSGI apps

You can also serve any app written in an
[ASGI](https://asgi.readthedocs.io/en/latest/) or
[WSGI](https://en.wikipedia.org/wiki/Web_Server_Gateway_Interface)-compatible
web framework on Modal.

ASGI provides support for async web frameworks. WSGI provides support for
synchronous web frameworks.

### ASGI apps - FastAPI, FastHTML, Starlette

For ASGI apps, you can create a function decorated with
[`@modal.asgi_app`](https://modal.com/docs/reference/modal.asgi_app) that returns a reference to
your web app:

```python
image = modal.Image.debian_slim().pip_install("fastapi[standard]")

@app.function(image=image)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Request

    web_app = FastAPI()

    @web_app.post("/echo")
    async def echo(request: Request):
        body = await request.json()
        return body

    return web_app
```

Now, as before, when you deploy this script as a Modal App, you get a URL for
your app that you can hit:

The `@modal.concurrent` decorator enables a single container
to process multiple inputs at once, taking advantage of the asynchronous
event loops in ASGI applications. See [this guide](https://modal.com/docs/guide/concurrent-inputs)
for details.

#### ASGI Lifespan

While we recommend using [`@modal.enter`](https://modal.com/docs/guide/lifecycle-functions#enter) for defining container lifecycle hooks, we also support the [ASGI lifespan protocol](https://asgi.readthedocs.io/en/latest/specs/lifespan.html). Lifespans begin when containers start, typically at the time of the first request. Here's an example using [FastAPI](https://fastapi.tiangolo.com/advanced/events/#lifespan):

```python
import modal

app = modal.App("fastapi-lifespan-app")

image = modal.Image.debian_slim().pip_install("fastapi[standard]")

@app.function(image=image)
@modal.asgi_app()
def fastapi_app_with_lifespan():
    from fastapi import FastAPI, Request

    def lifespan(wapp: FastAPI):
        print("Starting")
        yield
        print("Shutting down")

    web_app = FastAPI(lifespan=lifespan)

    @web_app.get("/")
    async def hello(request: Request):
        return "hello"

    return web_app
```

### WSGI apps - Django, Flask

You can serve WSGI apps using the
[`@modal.wsgi_app`](https://modal.com/docs/reference/modal.wsgi_app) decorator:

```python
image = modal.Image.debian_slim().pip_install("flask")

@app.function(image=image)
@modal.concurrent(max_inputs=100)
@modal.wsgi_app()
def flask_app():
    from flask import Flask, request

    web_app = Flask(__name__)

    @web_app.post("/echo")
    def echo():
        return request.json

    return web_app
```

See [Flask's docs](https://flask.palletsprojects.com/en/2.1.x/deploying/asgi/)
for more information on using Flask as a WSGI app.

Because WSGI apps are synchronous, concurrent inputs will be run on separate
threads. See [this guide](https://modal.com/docs/guide/concurrent-inputs) for details.

## Non-ASGI web servers

Not all web frameworks offer an ASGI or WSGI interface. For example,
[`aiohttp`](https://docs.aiohttp.org/) and [`tornado`](https://www.tornadoweb.org/)
use their own asynchronous network binding, while others like
[`text-generation-inference`](https://github.com/huggingface/text-generation-inference)
actually expose a Rust-based HTTP server running as a subprocess.

For these cases, you can use the
[`@modal.web_server`](https://modal.com/docs/reference/modal.web_server) decorator to "expose" a
port on the container:

```python
@app.function()
@modal.concurrent(max_inputs=100)
@modal.web_server(8000)
def my_file_server():
    import subprocess
    subprocess.Popen("python -m http.server -d / 8000", shell=True)
```

Just like all web endpoints on Modal, this is only run on-demand. The function
is executed on container startup, creating a file server at the root directory.
When you hit the web endpoint URL, your request will be routed to the file
server listening on port `8000`.

For `@web_server` endpoints, you need to make sure that the application binds to
the external network interface, not just localhost. This usually means binding
to `0.0.0.0` instead of `127.0.0.1`.

See our examples of how to serve [Streamlit](https://modal.com/docs/examples/serve_streamlit) and
[ComfyUI](https://modal.com/docs/examples/comfyapp) on Modal.

## Serve many configurations with parametrized functions

Python functions that launch ASGI/WSGI apps or web servers on Modal
cannot take arguments.

One simple pattern for allowing client-side configuration of these web endpoints
is to use [parametrized functions](https://modal.com/docs/guide/parametrized-functions).
Each different choice for the values of the parameters will create a distinct
auto-scaling container pool.

```python
@app.cls()
@modal.concurrent(max_inputs=100)
class Server:
    root: str = modal.parameter(default=".")

    @modal.web_server(8000)
    def files(self):
        import subprocess
        subprocess.Popen(f"python -m http.server -d {self.root} 8000", shell=True)
```

The values are provided in URLs as query parameters:

```bash
curl https://ecorp--server-files.modal.run		# use the default value
curl https://ecorp--server-files.modal.run?root=.cache  # use a different value
curl https://ecorp--server-files.modal.run?root=%2F	# don't forget to URL encode!
```

For details, see [this guide to parametrized functions](https://modal.com/docs/guide/parametrized-functions).

## WebSockets

Functions annotated with `@web_server`, `@asgi_app`, or `@wsgi_app` also support
the WebSocket protocol. Consult your web framework for appropriate documentation
on how to use WebSockets with that library.

WebSockets on Modal maintain a single function call per connection, which can be
useful for keeping state around. Most of the time, you will want to set your
handler function to [allow concurrent inputs](https://modal.com/docs/guide/concurrent-inputs),
which allows multiple simultaneous WebSocket connections to be handled by the
same container.

We support the full WebSocket protocol as per
[RFC 6455](https://www.rfc-editor.org/rfc/rfc6455), but we do not yet have
support for [RFC 8441](https://www.rfc-editor.org/rfc/rfc8441) (WebSockets over
HTTP/2) or [RFC 7692](https://datatracker.ietf.org/doc/html/rfc7692)
(`permessage-deflate` extension). WebSocket messages can be up to 2 MiB each.

## Performance and scaling

If you have no active containers when the web endpoint receives a request, it will
experience a "cold start". Consult the guide page on
[cold start performance](https://modal.com/docs/guide/cold-start) for more information on when
Functions will cold start and advice how to mitigate the impact.

If your Function uses `@modal.concurrent`, multiple requests to the same
endpoint may be handled by the same container. Beyond this limit, additional
containers will start up to scale your App horizontally. When you reach the
Function's limit on containers, requests will queue for handling.

Each workspace on Modal has a rate limit on total operations. For a new account,
this is set to 200 function inputs or web endpoint requests per second, with a
burst multiplier of 5 seconds. If you reach the rate limit, excess requests to
web endpoints will return a
[429 status code](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/429),
and you'll need to [get in touch](mailto:support@modal.com) with us about
raising the limit.

Web endpoint request bodies can be up to 4 GiB, and their response bodies are
unlimited in size.

## Authentication

Modal offers first-class web endpoint protection via [proxy auth tokens](https://modal.com/docs/guide/webhook-proxy-auth).
Proxy auth tokens protect web endpoints by requiring a key and token combination to be passed
in the `Modal-Key` and `Modal-Secret` headers.
Modal works as a proxy, rejecting requests that aren't authorized to access
your endpoint.

We also support standard techniques for securing web servers.

### Token-based authentication

This is easy to implement in whichever framework you're using. For example, if
you're using `@modal.fastapi_endpoint` or `@modal.asgi_app` with FastAPI, you
can validate a Bearer token like this:

```python
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import modal

image = modal.Image.debian_slim().pip_install("fastapi[standard]")
app = modal.App("auth-example", image=image)

auth_scheme = HTTPBearer()

@app.function(secrets=[modal.Secret.from_name("my-web-auth-token")])
@modal.fastapi_endpoint()
async def f(request: Request, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    import os

    print(os.environ["AUTH_TOKEN"])

    if token.credentials != os.environ["AUTH_TOKEN"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Function body
    return "success!"
```

This assumes you have a [Modal Secret](https://modal.com/secrets) named
`my-web-auth-token` created, with contents `{AUTH_TOKEN: secret-random-token}`.
Now, your endpoint will return a 401 status code except when you hit it with the
correct `Authorization` header set (note that you have to prefix the token with
`Bearer `):

```bash
curl --header "Authorization: Bearer secret-random-token" https://modal-labs--auth-example-f.modal.run
```

### Client IP address

You can access the IP address of the client making the request. This can be used
for geolocation, whitelists, blacklists, and rate limits.

```python
from fastapi import Request

import modal

image = modal.Image.debian_slim().pip_install("fastapi[standard]")
app = modal.App(image=image)

@app.function()
@modal.fastapi_endpoint()
def get_ip_address(request: Request):
    return f"Your IP address is {request.client.host}"
```

#### Streaming endpoints

# Streaming endpoints

Modal web endpoints support streaming responses using FastAPI's
[`StreamingResponse`](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
class. This class accepts asynchronous generators, synchronous generators, or
any Python object that implements the
[_iterator protocol_](https://docs.python.org/3/library/stdtypes.html#typeiter),
and can be used with Modal Functions!

## Simple example

This simple example combines Modal's `@modal.fastapi_endpoint` decorator with a
`StreamingResponse` object to produce a real-time SSE response.

```python
import time

def fake_event_streamer():
    for i in range(10):
        yield f"data: some data {i}\n\n".encode()
        time.sleep(0.5)

@app.function(image=modal.Image.debian_slim().pip_install("fastapi[standard]"))
@modal.fastapi_endpoint()
def stream_me():
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        fake_event_streamer(), media_type="text/event-stream"
    )
```

If you serve this web endpoint and hit it with `curl`, you will see the ten SSE
events progressively appear in your terminal over a ~5 second period.

```shell
curl --no-buffer https://modal-labs--example-streaming-stream-me.modal.run
```

The MIME type of `text/event-stream` is important in this example, as it tells
the downstream web server to return responses immediately, rather than buffering
them in byte chunks (which is more efficient for compression).

You can still return other content types like large files in streams, but they
are not guaranteed to arrive as real-time events.

## Streaming responses with `.remote`

A Modal Function wrapping a generator function body can have its response passed
directly into a `StreamingResponse`. This is particularly useful if you want to
do some GPU processing in one Modal Function that is called by a CPU-based web
endpoint Modal Function.

```python
@app.function(gpu="any")
def fake_video_render():
    for i in range(10):
        yield f"data: finished processing some data from GPU {i}\n\n".encode()
        time.sleep(1)

@app.function(image=modal.Image.debian_slim().pip_install("fastapi[standard]"))
@modal.fastapi_endpoint()
def hook():
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        fake_video_render.remote_gen(), media_type="text/event-stream"
    )
```

## Streaming responses with `.map` and `.starmap`

You can also combine Modal Function parallelization with streaming responses,
enabling applications to service a request by farming out to dozens of
containers and iteratively returning result chunks to the client.

```python
@app.function()
def map_me(i):
    return f"segment {i}\n"

@app.function(image=modal.Image.debian_slim().pip_install("fastapi[standard]"))
@modal.fastapi_endpoint()
def mapped():
    from fastapi.responses import StreamingResponse
    return StreamingResponse(map_me.map(range(10)), media_type="text/plain")
```

This snippet will spread the ten `map_me(i)` executions across containers, and
return each string response part as it completes. By default the results will be
ordered, but if this isn't necessary pass `order_outputs=False` as keyword
argument to the `.map` call.

### Asynchronous streaming

The example above uses a synchronous generator, which automatically runs on its
own thread, but in asynchronous applications, a loop over a `.map` or `.starmap`
call can block the event loop. This will stop the `StreamingResponse` from
returning response parts iteratively to the client.

To avoid this, you can use the `.aio()` method to convert a synchronous `.map`
into its async version. Also, other blocking calls should be offloaded to a
separate thread with `asyncio.to_thread()`. For example:

```python
@app.function(gpu="any", image=modal.Image.debian_slim().pip_install("fastapi[standard]"))
@modal.fastapi_endpoint()
async def transcribe_video(request):
    from fastapi.responses import StreamingResponse

    segments = await asyncio.to_thread(split_video, request)
    return StreamingResponse(wrapper(segments), media_type="text/event-stream")

# Notice that this is an async generator.
async def wrapper(segments):
    async for partial_result in transcribe_video.map.aio(segments):
        yield "data: " + partial_result + "\n\n"
```

## Further examples

- Complete code the for the simple examples given above is available
  [in our modal-examples Github repository](https://github.com/modal-labs/modal-examples/blob/main/07_web_endpoints/streaming.py).
- [An end-to-end example of streaming Youtube video transcriptions with OpenAI's whisper model.](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/openai_whisper/streaming/main.py)

#### Web endpoint URLs

# Web endpoint URLs

This guide documents the behavior of URLs for [web endpoints](https://modal.com/docs/guide/webhooks)
on Modal: automatic generation, configuration, programmatic retrieval, and more.

## Determine the URL of a web endpoint from code

Modal Functions with the
[`fastapi_endpoint`](https://modal.com/docs/reference/modal.fastapi_endpoint),
[`asgi_app`](https://modal.com/docs/reference/modal.asgi_app),
[`wsgi_app`](https://modal.com/docs/reference/modal.wsgi_app),
or [`web_server`](https://modal.com/docs/reference/modal.web_server) decorator
are made available over the Internet when they are
[`serve`d](https://modal.com/docs/reference/cli/serve) or [`deploy`ed](https://modal.com/docs/reference/cli/deploy)
and so they have a URL.

This URL is displayed in the `modal` CLI output
and is available in the Modal [dashboard](https://modal.com/apps) for the Function.

To determine a Function's URL programmatically,
check its [`get_web_url()`](https://modal.com/docs/reference/modal.Function#get_web_url)
property:

```python
@app.function(image=modal.Image.debian_slim().pip_install("fastapi[standard]"))
@modal.fastapi_endpoint(docs=True)
def show_url() -> str:
    return show_url.get_web_url()
```

For deployed Functions, this also works from other Python code!
You just need to do a [`from_name`](https://modal.com/docs/reference/modal.Function#from_name)
based on the name of the Function and its [App](https://modal.com/docs/guide/apps):

```python notest
import requests

remote_function = modal.Function.from_name("app", "show_url")
remote_function.get_web_url() == requests.get(handle.get_web_url()).json()
```

## Auto-generated URLs

By default, Modal Functions
will be served from the `modal.run` domain.
The full URL will be constructed from a number of pieces of information
to uniquely identify the endpoint.

At a high-level, web endpoint URLs for deployed applications have the
following structure: `https://<source>--<label>.modal.run`.

The `source` component represents the workspace and environment where the App is
deployed. If your workspace has only a single environment, the `source` will
just be the workspace name. Multiple environments are disambiguated by an
["environment suffix"](https://modal.com/docs/guide/environments#environment-web-suffixes), so
the full source would be `<workspace>-<suffix>`. However, one environment per
workspace is allowed to have a null suffix, in which case the source would just
be `<workspace>`.

The `label` component represents the specific App and Function that the endpoint
routes to. By default, these are concatenated with a hyphen, so the label would
be `<app>-<function>`.

These components are normalized to contain only lowercase letters, numerals, and dashes.

To put this all together, consider the following example. If a member of the
`ECorp` workspace uses the `main` environment (which has `prod` as its web
suffix) to deploy the `text_to_speech` app with a webhook for the `flask-app`
function, the URL will have the following components:

- _Source_:
  - _Workspace name slug_: `ECorp` → `ecorp`
  - _Environment web suffix slug_: `main` → `prod`
- _Label_:
  - _App name slug_: `text_to_speech` → `text-to-speech`
  - _Function name slug_: `flask_app` → `flask-app`

The full URL will be `https://ecorp-prod--text-to-speech-flask-app.modal.run`.

## User-specified labels

It's also possible to customize the `label` used for each Function
by passing a parameter to the relevant endpoint decorator:

```python
import modal

image = modal.Image.debian_slim().pip_install("fastapi")
app = modal.App(name="text_to_speech", image=image)

@app.function()
@modal.fastapi_endpoint(label="speechify")
def web_endpoint_handler():
    ...
```

Building on the example above, this code would produce the following URL:
`https://ecorp-prod--speechify.modal.run`.

User-specified labels are not automatically normalized, but labels with
invalid characters will be rejected.

## Ephemeral apps

To support development workflows, webhooks for ephemeral apps (i.e., apps
created with `modal serve`) will have a `-dev` suffix appended to their URL
label (regardless of whether the label is auto-generated or user-specified).
This prevents development work from interfering with deployed versions of the
same app.

If an ephemeral app is serving a webhook while another ephemeral webhook is
created seeking the same web endpoint label, the new function will _steal_ the
running webhook's label.

This ensures that the latest iteration of the ephemeral function is
serving requests and that older ones stop receiving web traffic.

## Truncation

If a generated subdomain label is longer than 63 characters, it will be
truncated.

For example, the following subdomain label is too long, 67 characters:
`ecorp--text-to-speech-really-really-realllly-long-function-name-dev`.

The truncation happens by calculating a SHA-256 hash of the overlong label, then
taking the first 6 characters of this hash. The overlong subdomain label is
truncated to 56 characters, and then joined by a dash to the hash prefix. In
the above example, the resulting URL would be
`ecorp--text-to-speech-really-really-rea-1b964b-dev.modal.run`.

The combination of the label hashing and truncation provides a unique list of 63
characters, complying with both DNS system limits and uniqueness requirements.

## Custom domains

**Custom domains are available on our
[Team and Enterprise plans](https://modal.com/settings/plans).**

For more customization, you can use your own domain names with Modal web
endpoints. If your [plan](https://modal.com/pricing) supports custom domains, visit the [Domains
tab](https://modal.com/settings/domains) in your workspace settings to add a domain name to your
workspace.

You can use three kinds of domains with Modal:

- **Apex:** root domain names like `example.com`
- **Subdomain:** single subdomain entries such as `my-app.example.com`,
  `api.example.com`, etc.
- **Wildcard domain:** either in a subdomain like `*.example.com`, or in a
  deeper level like `*.modal.example.com`

You'll be asked to update your domain DNS records with your domain name
registrar and then validate the configuration in Modal. Once the records have
been properly updated and propagated, your custom domain will be ready to use.

You can assign any Modal web endpoint to any registered domain in your workspace
with the `custom_domains` argument.

```python
import modal

app = modal.App("custom-domains-example")

@app.function()
@modal.fastapi_endpoint(custom_domains=["api.example.com"])
def hello(message: str):
    return {"message": f"hello {message}"}
```

You can then run `modal deploy` to put your web endpoint online, live.

```shell
$ curl -s https://api.example.com?message=world
{"message": "hello world"}
```

Note that Modal automatically generates and renews TLS certificates for your
custom domains. Since we do this when your domain is first accessed, there may
be an additional 1-2s latency on the first request. Additional requests use a
cached certificate.

You can also register multiple domain names and associate them with the same web
endpoint.

```python
import modal

app = modal.App("custom-domains-example-2")

@app.function()
@modal.fastapi_endpoint(custom_domains=["api.example.com", "api.example.net"])
def hello(message: str):
    return {"message": f"hello {message}"}
```

For **Wildcard** domains, Modal will automatically resolve arbitrary custom
endpoints (and issue TLS certificates). For example, if you add the wildcard
domain `*.example.com`, then you can create any custom domains under
`example.com`:

```python
import random
import modal

app = modal.App("custom-domains-example-2")

random_domain_name = random.choice(range(10))

@app.function()
@modal.fastapi_endpoint(custom_domains=[f"{random_domain_name}.example.com"])
def hello(message: str):
    return {"message": f"hello {message}"}
```

Custom domains can also be used with
[ASGI](https://modal.com/docs/reference/modal.asgi_app#modalasgi_app) or
[WSGI](https://modal.com/docs/reference/modal.wsgi_app) apps using the same
`custom_domains` argument.

#### Request timeouts

# Request timeouts

Web endpoint (a.k.a. webhook) requests should complete quickly, ideally within a
few seconds. All web endpoint function types
([`web_endpoint`, `asgi_app`, `wsgi_app`](https://modal.com/docs/reference/modal.web_endpoint))
have a maximum HTTP request timeout of 150 seconds enforced. However, the
underlying Modal function can have a longer [timeout](https://modal.com/docs/guide/timeouts).

In case the function takes more than 150 seconds to complete, a HTTP status 303
redirect response is returned pointing at the original URL with a special query
parameter linking it that request. This is the _result URL_ for your function.
Most web browsers allow for up to 20 such redirects, effectively allowing up to
50 minutes (20 \* 150 s) for web endpoints before the request times out.

(**Note:** This does not work with requests that require
[CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS), since the
response will not have been returned from your code in time for the server to
populate CORS headers.)

Some libraries and tools might require you to add a flag or option in order to
follow redirects automatically, e.g. `curl -L ...` or `http --follow ...`.

The _result URL_ can be reloaded without triggering a new request. It will block
until the request completes.

(**Note:** As of March 2025, the Python standard library's `urllib` module has the
maximum number of redirects to any single URL set to 4 by default ([source](https://github.com/python/cpython/blob/main/Lib/urllib/request.py)), which would limit the total timeout to 12.5 minutes (5 \* 150 s = 750 s) unless this setting is overridden.)

## Polling solutions

Sometimes it can be useful to be able to poll for results rather than wait for a
long running HTTP request. The easiest way to do this is to have your web
endpoint spawn a `modal.Function` call and return the function call id that
another endpoint can use to poll the submitted function's status. Here is an
example:

```python
import fastapi

import modal

image = modal.Image.debian_slim().pip_install("fastapi[standard]")
app = modal.App(image=image)

web_app = fastapi.FastAPI()

@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app

@app.function()
def slow_operation():
    ...

@web_app.post("/accept")
async def accept_job(request: fastapi.Request):
    call = slow_operation.spawn()
    return {"call_id": call.object_id}

@web_app.get("/result/{call_id}")
async def poll_results(call_id: str):
    function_call = modal.FunctionCall.from_id(call_id)
    try:
        return function_call.get(timeout=0)
    except TimeoutError:
        http_accepted_code = 202
        return fastapi.responses.JSONResponse({}, status_code=http_accepted_code)
```

[_Document OCR Web App_](https://modal.com/docs/examples/doc_ocr_webapp) is an example that uses
this pattern.

#### Proxy Auth Tokens

# Proxy Auth Tokens

Use Proxy Auth Tokens to prevent unauthorized clients from triggering your web endpoints.

```python
import modal

image = modal.Image.debian_slim().pip_install("fastapi")
app = modal.App("proxy-auth-public", image=image)

@app.function()
@modal.fastapi_endpoint()
def public():
    return "hello world"

@app.function()
@modal.fastapi_endpoint(requires_proxy_auth=True)
def private():
    return "hello friend"
```

The `public` endpoint can be hit by any client over the Internet.

```bash
curl https://public-url--goes-here.modal.run
```

The `private` endpoint cannot.

```bash
curl --fail-with-body https://private-url--goes-here.modal.run
# modal-http: missing credentials for proxy authorization
# curl: (22) The requested URL returned error: 401
# https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/401
```

Authorization is demonstrated via a Proxy Auth Token. You can create a Proxy Auth Token for your workspace [here](https://modal.com/settings/proxy-auth-tokens).
In requests to the web endpoint, clients supply the Token ID and Token Secret in the `Modal-Key` and `Modal-Secret` HTTP headers.

```bash
export TOKEN_ID=wk-1234abcd
export TOKEN_SECRET=ws-1234abcd
curl -H "Modal-Key: $TOKEN_ID" \
     -H "Modal-Secret: $TOKEN_SECRET" \
     https://private-url--goes-here.modal.run
```

Proxy authorization can be added to [web endpoints](https://modal.com/docs/guide/webhooks) created by the
[`fastapi_endpoint`](https://modal.com/docs/reference/modal.fastapi_endpoint),
[`asgi_app`](https://modal.com/docs/reference/modal.asgi_app),
[`wsgi_app`](https://modal.com/docs/reference/modal.wsgi_app), or
[`web_server`](https://modal.com/docs/reference/modal.web_server) decorators,
which are otherwise publicly available.

Everyone within the workspace of the web endpoint can manage its Proxy Auth Tokens.

### Networking

#### Tunnels

# Tunnels

Modal allows you to expose live TCP ports on a Modal container. This is done by
creating a _tunnel_ that forwards the port to the public Internet.

```python
import modal

app = modal.App()

@app.function()
def start_app():
    # Inside this `with` block, port 8000 on the container can be accessed by
    # the address at `tunnel.url`, which is randomly assigned.
    with modal.forward(8000) as tunnel:
        print(f"tunnel.url        = {tunnel.url}")
        print(f"tunnel.tls_socket = {tunnel.tls_socket}")
        # ... start some web server at port 8000, using any framework
```

Tunnels are direct connections and terminate TLS automatically. Within a few
milliseconds of container startup, this function prints a message such as:

```
tunnel.url        = https://wtqcahqwhd4tu0.r5.modal.host
tunnel.tls_socket = ('wtqcahqwhd4tu0.r5.modal.host', 443)
```

You can also create tunnels on a [Sandbox](https://modal.com/docs/guide/sandbox-networking#forwarding-ports)
to directly expose the container's ports.

## Build with tunnels

Tunnels are the fastest way to get a low-latency, direct connection to a running
container. You can use them to run live browser applications with **interactive
terminals**, **Jupyter notebooks**, **VS Code servers**, and more.

As a quick example, here is how you would expose a Jupyter notebook:

```python
import os
import secrets
import subprocess

import modal

image = modal.Image.debian_slim().pip_install("jupyterlab")
app = modal.App(image=image)

@app.function()
def run_jupyter():
    token = secrets.token_urlsafe(13)
    with modal.forward(8888) as tunnel:
        url = tunnel.url + "/?token=" + token
        print(f"Starting Jupyter at {url}")
        subprocess.run(
            [
                "jupyter",
                "lab",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                "--port=8888",
                "--LabApp.allow_origin='*'",
                "--LabApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": token, "SHELL": "/bin/bash"},
            stderr=subprocess.DEVNULL,
        )
```

When you run the function, it starts Jupyter and gives you the public URL. It's
as simple as that.

All Modal features are supported. If you
[need GPUs](https://modal.com/docs/guide/gpu), pass `gpu=` to the
`@app.function()` decorator. If you
[need more CPUs, RAM](https://modal.com/docs/guide/resources), or to attach
[volumes](https://modal.com/docs/guide/volumes), those
also just work.

### Programmable startup

The tunnel API is completely on-demand, so you can start them as the result of a
web request.

For example, you could make something like Jupyter Hub without leaving Modal,
giving your users their own Jupyter notebooks when they visit a URL:

```python
import modal

image = modal.Image.debian_slim().pip_install("fastapi[standard]")
app = modal.App(image=image)

@app.function(timeout=900)  # 15 minutes
def run_jupyter(q):
    ...  # as before, but return the URL on app.q

@app.function()
@modal.fastapi_endpoint(method="POST")
def jupyter_hub():
    from fastapi import HTTPException
    from fastapi.responses import RedirectResponse

    ...  # do some validation on the secret or bearer token

    if is_valid:
        with modal.Queue.ephemeral() as q:
            run_jupyter.spawn(q)
            url = q.get()
            return RedirectResponse(url, status_code=303)

    else:
        raise HTTPException(401, "Not authenticated")
```

This gives every user who sends a POST request to the web endpoint their own
Jupyter notebook server, on a fully isolated Modal container.

You could do the same with VS Code and get some basic version of an instant,
serverless IDE!

### Advanced: Unencrypted TCP tunnels

By default, tunnels are only exposed to the Internet at a secure random URL, and
connections have automatic TLS (the "S" in HTTPS). However, sometimes you might
need to expose a protocol like an SSH server that goes directly over TCP. In
this case, we have support for _unencrypted_ tunnels:

```python notest
with modal.forward(8000, unencrypted=True) as tunnel:
    print(f"tunnel.tcp_socket = {tunnel.tcp_socket}")
```

Might produce an output like:

```
tunnel.tcp_socket = ('r3.modal.host', 23447)
```

You can then connect over TCP, for example with `nc r3.modal.host 23447`. Unlike
encrypted TLS sockets, these cannot be given a non-guessable, cryptographically
random URL due to how the TCP protocol works, so they are assigned a random port
number instead.

## Pricing

Modal only charges for containers based on
[the resources you use](https://modal.com/pricing). There is no additional
charge for having an active tunnel.

For example, if you start a Jupyter notebook on port 8888 and access it via
tunnel, you can use it for an hour for development (with 0.01 CPUs) and then
actually run an intensive job with 16 CPUs for one minute. The amount you would
be billed for in that hour is 0.01 + 16 \* (1/60) = **0.28 CPUs**, even though
you had access to 16 CPUs without needing to restart your notebook.

## Security

Tunnels are run on Modal's private global network of Internet relays. On
startup, your container will connect to the nearest tunnel so you get the
minimum latency, very similar in performance to a direct connection with the
machine.

This makes them ideal for live debugging sessions, using web-based terminals
like [ttyd](https://github.com/tsl0922/ttyd).

The generated URLs are cryptographically random, but they are also public on the
Internet, so anyone can access your application if they are given the URL.

We do not currently do any detection of requests above L4, so if you are running
a web server, we will not add special proxy HTTP headers or translate HTTP/2.
You're just getting the TLS-encrypted TCP stream directly!

#### Proxies (beta)

# Proxies (beta)

You can securely connect with resources in your private network
using a Modal Proxy. Proxies are a secure tunnel between
Apps and exit nodes with static IPs. You can allow-list those static IPs
in your network firewall, making sure that only traffic originating from these
IP addresses is allowed into your network.

Proxies are unique and not shared between workspaces. All traffic
between your Apps and the Proxy server is encrypted using
[WireGuard](https://www.wireguard.com/).

Modal Proxies are built on top of [vprox](https://github.com/modal-labs/vprox),
a Modal open-source project used to create highly available proxy servers
using WireGuard.

_Modal Proxies are in beta. Please let us know if you run into issues._

## Creating a Proxy

Proxies are available for [Team Plan](https://modal.com/pricing) or [Enterprise](https://modal.com/pricing) users.

You can create Proxies in your workspace [Settings](https://modal.com/settings) page.
Team Plan users can create one Proxy and Enterprise users three Proxies. Each Proxy
can have a maximum of five static IP addresses.

Please reach out to [support@modal.com](mailto:support@modal.com) if you need greater limits.

## Using a Proxy

After a Proxy is online, add it to a Modal Function with the argument
`proxy=Proxy.from_name("<your-proxy>")`. For example:

```python
import modal
import subprocess

app = modal.App(image=modal.Image.debian_slim().apt_install("curl"))

@app.function(proxy=modal.Proxy.from_name("<your-proxy>"))
def my_ip():
    subprocess.run(["curl", "-s", "ifconfig.me"])

@app.local_entrypoint()
def main():
    my_ip.remote()
```

All network traffic from your Function will now use the Proxy as a tunnel.

The program above will always print the same IP address independent
of where it runs in Modal's infrastructure. If that same program
were to run without a Proxy, it would print a different IP
address depending on where it runs.

## Proxy performance

All traffic that goes through a Proxy is encrypted by WireGuard. This adds
latency to your Function's networking. If are experiencing networking issues
with Proxies related to performance, first add more IP addresses to your
Proxy (see [Adding more IP addresses a Proxy](#adding-more-ip-addresses-to-a-proxy)).

## Adding more IP addresses to a Proxy

Proxies support up to five static IP addresses. Adding IP addresses improves
throughput linearly.

You can add an IP address to your workspace in [Settings](https://modal.com/settings) > Proxies.
Select the desired Proxy and add a new IP.

If a Proxy has multiple IPs, Modal will randomly pick one when running your Function.

## Proxies and Sandboxes

Proxies can also be used with [Sandboxes](https://modal.com/docs/guide/sandbox). For example:

```python notest
import modal

app = modal.App.lookup("sandbox-proxy", create_if_missing=True)
sb = modal.Sandbox.create(
    app=app,
    image=modal.Image.debian_slim().apt_install("curl"),
    proxy=modal.Proxy.from_name("<your-proxy>"))

process = sb.exec("curl", "-s", "https://ifconfig.me")
stdout = process.stdout.read()
print(stdout)

sb.terminate()
```

Similarly to our Function implementation, this Sandbox program will
always print the same IP address.

#### Cluster networking

# Cluster networking

i6pn (IPv6 private networking) is Modal’s private container-to-container networking solution. It allows users to create clusters of Modal containers which can send network traffic to each other with low latency and high bandwidth (≥ 50Gbps).

Normally, `modal.Function` containers can initiate outbound network connections to the internet but they are not directly addressable by other containers. i6pn-enabled containers, on the other hand, can be directly connected to by other i6pn-enabled containers and this is a key enabler of Modal’s preview `@modal.experimental.clustered` functionality.

You can enable i6pn on any `modal.Function`:

```python
@app.function(i6pn=True)
def hello_private_network():
    import socket

    i6pn_addr = socket.getaddrinfo("i6pn.modal.local", None, socket.AF_INET6)[0][4][0]
    print(i6pn_addr) # fdaa:5137:3ebf:a70:1b9d:3a11:71f2:5f0f
```

In this snippet we see that the i6pn-enabled container is able to retrieve its own IPv6 address by
resolving `i6pn.modal.local`. For this Function container to discover the addresses of _other_ containers,
address sharing must be implemented using an auxiliary data structure, such as a shared `modal.Dict` or `modal.Queue`.

## Private networking

All i6pn network traffic is _Workspace private_.

![i6pn-diagram](https://modal-cdn.com/cdnbot/i6pn-1eksk4vuy_c4c4a0df.webp)

In the image above, Workspace A has subnet `fdaa:1::/48`, while Workspace B has subnet `fdaa:2::/48`.

You’ll notice they share the first 16 bits. This is because the `fdaa::/16` prefix contains all of our private network IPv6 addresses, while each workspace is assigned a random 32-bit identifier when it is created. Together, these form the 48-bit subnet.

The upshot of this is that only containers in the same workspace can see each other and send each other network packets. i6pn networking is secure by default.

## Region boundaries

Modal operates a [global fleet](https://modal.com/docs/guide/region-selection) and allows containers to run on multiple cloud providers and in many regions. i6pn networking is however region-scoped functionality, meaning that only i6pn-enabled containers in the same region can perform network communication.

Modal’s i6pn-enabled primitives such as `@modal.experimental.clustered` automatically restrict container geographic placement and cloud placement to ensure inter-container connectivity.

## Public network access to cluster networking

For cluster networked containers that need to be publicly accessible, you need to expose ports with [modal.Tunnel](https://modal.com/docs/guide/tunnels) because i6pn addresses are not publicly exposed.

Consider having a container setup a Tunnel and act as the gateway to the private cluster networking.

### Data sharing and storage

#### Passing local data

# Passing local data

If you have a function that needs access to some data not present in your Python
files themselves you have a few options for bundling that data with your Modal
app.

## Passing function arguments

The simplest and most straight-forward way is to read the data from your local
script and pass the data to the outermost Modal function call:

```python
import json

@app.function()
def foo(a):
    print(sum(a["numbers"]))

@app.local_entrypoint()
def main():
    data_structure = json.load(open("blob.json"))
    foo.remote(data_structure)
```

Any data of reasonable size that is serializable through
[cloudpickle](https://github.com/cloudpipe/cloudpickle) is passable as an
argument to Modal functions.

Refer to the section on [global variables](https://modal.com/docs/guide/global-variables) for how
to work with objects in global scope that can only be initialized locally.

## Including local files

For including local files for your Modal Functions to access, see [Defining Images](https://modal.com/docs/guide/images).

#### Volumes

# Volumes

Modal Volumes provide a high-performance distributed file system for your Modal applications.
They are designed for write-once, read-many I/O workloads, like creating machine learning model
weights and distributing them for inference.

This page is a high-level guide to using Modal Volumes.
For reference documentation on the `modal.Volume` object, see
[this page](https://modal.com/docs/reference/modal.Volume).
For reference documentation on the `modal volume` CLI command, see
[this page](https://modal.com/docs/reference/cli/volume).

## Volumes v2

A new generation of the file system, Volumes v2, is now available as a
beta preview.

> 🌱 Instructions that are specific to v2 Volumes will be annotated with 🌱
> below.

Read more about [Volumes v2](#volumes-v2-overview) below.

## Creating a Volume

The easiest way to create a Volume and use it as a part of your App is to use
the [`modal volume create`](https://modal.com/docs/reference/cli/volume#modal-volume-create) CLI command. This will create the Volume and output
some sample code:

```bash
% modal volume create my-volume
Created volume 'my-volume' in environment 'main'.
```

> 🌱 To create a v2 Volume, pass `--version=2` in the command above.

## Using a Volume on Modal

To attach an existing Volume to a Modal Function, use [`Volume.from_name`](https://modal.com/docs/reference/modal.Volume#from_name):

```python
vol = modal.Volume.from_name("my-volume")

@app.function(volumes={"/data": vol})
def run():
    with open("/data/xyz.txt", "w") as f:
        f.write("hello")
    vol.commit()  # Needed to make sure all changes are persisted before exit
```

You can also browse and manipulate Volumes from an ad hoc Modal Shell:

```bash
% modal shell --volume my-volume --volume another-volume
```

Volumes will be mounted under `/mnt`.

## Downloading a file from a Volume

While there’s no file size limit for individual files in a volume, the frontend only supports downloading files up to 16 MB. For larger files, please use the CLI:

```bash
% modal volume get my-volume xyz.txt xyz-local.txt
```

### Creating Volumes lazily from code

You can also create Volumes lazily from code using:

```python
vol = modal.Volume.from_name("my-volume", create_if_missing=True)
```

> 🌱 To create a v2 Volume, pass `version=2` to the call to `from_name()` in the code above.

This will create the Volume if it doesn't exist.

## Using a Volume from outside of Modal

Volumes can also be used outside Modal via the [Python SDK](https://modal.com/docs/reference/modal.Volume#modalvolume) or our [CLI](https://modal.com/docs/reference/cli/volume).

### Using a Volume from local code

You can interact with Volumes from anywhere you like using the `modal` Python client library.

```python notest
vol = modal.Volume.from_name("my-volume")

with vol.batch_upload() as batch:
    batch.put_file("local-path.txt", "/remote-path.txt")
    batch.put_directory("/local/directory/", "/remote/directory")
    batch.put_file(io.BytesIO(b"some data"), "/foobar")
```

For more details, see the [reference documentation](https://modal.com/docs/reference/modal.Volume).

### Using a Volume via the command line

You can also interact with Volumes using the command line interface. You can run
`modal volume` to get a full list of its subcommands:

```bash
% modal volume
Usage: modal volume [OPTIONS] COMMAND [ARGS]...

 Read and edit modal.Volume volumes.
 Note: users of modal.NetworkFileSystem should use the modal nfs command instead.

╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                                                                            │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ File operations ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ cp       Copy within a modal.Volume. Copy source file to destination file or multiple source files to destination directory.                                                                           │
│ get      Download files from a modal.Volume object.                                                                                                                                                    │
│ ls       List files and directories in a modal.Volume volume.                                                                                                                                          │
│ put      Upload a file or directory to a modal.Volume.                                                                                                                                                 │
│ rm       Delete a file or directory from a modal.Volume.                                                                                                                                               │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Management ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ create   Create a named, persistent modal.Volume.                                                                                                                                                      │
│ delete   Delete a named, persistent modal.Volume.                                                                                                                                                      │
│ list     List the details of all modal.Volume volumes in an Environment.                                                                                                                               │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

For more details, see the [reference documentation](https://modal.com/docs/reference/cli/volume).

## Volume commits and reloads

Unlike a normal filesystem, you need to explicitly reload the Volume to see
changes made since it was first mounted. This reload is handled by invoking the
[`.reload()`](https://modal.com/docs/reference/modal.Volume#reload) method on a Volume object.
Similarly, any Volume changes made within a container need to be committed for
those the changes to become visible outside the current container. This is handled
periodically by [background commits](#background-commits) and directly by invoking
the [`.commit()`](https://modal.com/docs/reference/modal.Volume#commit)
method on a `modal.Volume` object.

At container creation time the latest state of an attached Volume is mounted. If
the Volume is then subsequently modified by a commit operation in another
running container, that Volume modification won't become available until the
original container does a [`.reload()`](https://modal.com/docs/reference/modal.Volume#reload).

Consider this example which demonstrates the effect of a reload:

```python
import pathlib
import modal

app = modal.App()

volume = modal.Volume.from_name("my-volume")

p = pathlib.Path("/root/foo/bar.txt")

@app.function(volumes={"/root/foo": volume})
def f():
    p.write_text("hello")
    print(f"Created {p=}")
    volume.commit()  # Persist changes
    print(f"Committed {p=}")

@app.function(volumes={"/root/foo": volume})
def g(reload: bool = False):
    if reload:
        volume.reload()  # Fetch latest changes
    if p.exists():
        print(f"{p=} contains '{p.read_text()}'")
    else:
        print(f"{p=} does not exist!")

@app.local_entrypoint()
def main():
    g.remote()  # 1. container for `g` starts
    f.remote()  # 2. container for `f` starts, commits file
    g.remote(reload=False)  # 3. reuses container for `g`, no reload
    g.remote(reload=True)   # 4. reuses container, but reloads to see file.
```

The output for this example is this:

```
p=PosixPath('/root/foo/bar.txt') does not exist!
Created p=PosixPath('/root/foo/bar.txt')
Committed p=PosixPath('/root/foo/bar.txt')
p=PosixPath('/root/foo/bar.txt') does not exist!
p=PosixPath('/root/foo/bar.txt') contains hello
```

This code runs two containers, one for `f` and one for `g`. Only the last
function invocation reads the file created and committed by `f` because it was
configured to reload.

### Background commits

Modal Volumes run background commits:
every few seconds while your Function executes,
the contents of attached Volumes will be committed
without your application code calling `.commit`.
A final snapshot and commit is also automatically performed on container shutdown.

Being able to persist changes to Volumes without changing your application code
is especially useful when [training or fine-tuning models using frameworks](#model-checkpointing).

## Model serving

A single ML model can be served by simply baking it into a `modal.Image` at
build time using [`run_function`](https://modal.com/docs/reference/modal.Image#run_function). But
if you have dozens of models to serve, or otherwise need to decouple image
builds from model storage and serving, use a `modal.Volume`.

Volumes can be used to save a large number of ML models and later serve any one
of them at runtime with great performance. This snippet below shows the
basic structure of the solution.

```python
import modal

app = modal.App()
volume = modal.Volume.from_name("model-store")
model_store_path = "/vol/models"

@app.function(volumes={model_store_path: volume}, gpu="any")
def run_training():
    model = train(...)
    save(model_store_path, model)
    volume.commit()  # Persist changes

@app.function(volumes={model_store_path: volume})
def inference(model_id: str, request):
    try:
        model = load_model(model_store_path, model_id)
    except NotFound:
        volume.reload()  # Fetch latest changes
        model = load_model(model_store_path, model_id)
    return model.run(request)
```

For more details, see our [guide to storing model weights on Modal](https://modal.com/docs/guide/model-weights).

## Model checkpointing

Checkpoints are snapshots of an ML model and can be configured by the callback
functions of ML frameworks. You can use saved checkpoints to restart a training
job from the last saved checkpoint. This is particularly helpful in managing
[preemption](https://modal.com/docs/guide/preemption).

For more, see our [example code for long-running training](https://modal.com/docs/examples/long-training).

### Hugging Face `transformers`

To periodically checkpoint into a `modal.Volume`, just set the `Trainer`'s
[`output_dir`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.output_dir)
to a directory in the Volume.

```python
import pathlib

volume = modal.Volume.from_name("my-volume")
VOL_MOUNT_PATH = pathlib.Path("/vol")

@app.function(
    gpu="A10G",
    timeout=2 * 60 * 60,  # run for at most two hours
    volumes={VOL_MOUNT_PATH: volume},
)
def finetune():
    from transformers import Seq2SeqTrainer
    ...

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(VOL_MOUNT_PATH / "model"),
        # ... more args here
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_xsum_train,
        eval_dataset=tokenized_xsum_test,
    )
```

## Volume performance

Volumes work best when they contain less than 50,000 files and directories. The
latency to attach or modify a Volume scales linearly with the number of files in
the Volume, and past a few tens of thousands of files the linear component
starts to dominate the fixed overhead.

There is currently a hard limit of 500,000 inodes (files, directories and
symbolic links) per Volume. If you reach this limit, any further attempts to
create new files or directories will error with
[`ENOSPC` (No space left on device)](https://pubs.opengroup.org/onlinepubs/9799919799/).

If you need to work with a large number of files, consider using Volumes v2!
It is currently in beta. See below for more info.

## Filesystem consistency

### Concurrent modification

Concurrent modification from multiple containers is supported, but concurrent
modifications of the same files should be avoided. Last write wins in case of
concurrent modification of the same file — any data the last writer didn't have
when committing changes will be lost!

The number of commits you can run concurrently is limited. If you run too many
concurrent commits each commit will take longer due to contention. If you are
committing small changes, avoid doing more than 5 concurrent commits (the number
of concurrent commits you can make is proportional to the size of the changes
being committed).

As a result, Volumes are typically not a good fit for use cases where you need
to make concurrent modifications to the same file (nor is distributed file
locking supported).

While a reload is in progress the Volume will appear empty to the container that
initiated the reload. That means you cannot read from or write to a Volume in a
container where a reload is ongoing (note that this only applies to the
container where the reload was issued, other containers remain unaffected).

### Busy Volume errors

You can only reload a Volume when there no open files on the Volume. If you have
open files on the Volume the [`.reload()`](https://modal.com/docs/reference/modal.Volume#reload)
operation will fail with "volume busy". The following is a simple example of how
a "volume busy" error can occur:

```python
volume = modal.Volume.from_name("my-volume")

@app.function(volumes={"/vol": volume})
def reload_with_open_files():
    f = open("/vol/data.txt", "r")
    volume.reload()  # Cannot reload when files in the Volume are open.
```

### Can't find file on Volume errors

When accessing files in your Volume, don't forget to pre-pend where your Volume
is mounted in the container.

In the example below, where the Volume has been mounted at `/data`, "hello" is
being written to `/data/xyz.txt`.

```python
import modal

app = modal.App()
vol = modal.Volume.from_name("my-volume")

@app.function(volumes={"/data": vol})
def run():
    with open("/data/xyz.txt", "w") as f:
        f.write("hello")
    vol.commit()
```

If you instead write to `/xyz.txt`, the file will be saved to the local disk of the Modal Function.
When you dump the contents of the Volume, you will not see the `xyz.txt` file.

## Volumes v2 overview

Volumes v2 generally behave just like Volumes v1, and most of the existing APIs
and CLI commands that you are used to will work the same between versions.
Because the file system implementation is completely different, there will be
some significant performance characteristics that can differ from version 1
Volumes. Below is an outline of the key differences you should be aware of.

### Volumes v2 is still in beta

This new file system version is still in beta, and we cannot guarantee that
no data will be lost. We don't recommend using Volumes v2 for any
mission-critical data at this time. You can still reap the benefits of v2 for
data that isn't precious, or that is easy to rebuild, such as log files,
regularly updated training data and model weights, caches, and more.

### Volumes v2 are HIPAA compliant

If you delete the volume, the data is be guaranteed to be lost according to HIPAA requirements.

### Volumes v2 is more scaleable

Volumes v2 support more files, higher throughput, and more irregular access
patterns. Commits and reloads are also faster.

Additionally, Volumes v2 supports hard-linking of files, where multiple paths
can point to the same inode.

### In v2, you can store as many files as you want

There is no limit on the number of files in Volumes v2.

By contrast, in Volumes v1, there is a limit on the number of files of 500,000,
and we recommend keeping the count to 50,000 or less.

### In v2, you can write concurrently from hundreds of containers

The file system should not experience any performance degradation as more
containers write to distinct files simultaneously.

By contrast, in Volumes v1, we recommend no more than five writers access the
Volume at once.

Note, however, that concurrent access to a particular _file_ in a Volume still
has last-write-wins semantics in many circumstances. These semantics are
unacceptable for most applications, so any particular file should only be
written to by a single container at a time.

### In v2, random accesses have improved performance

In v1, writes to locations inside a file would sometimes incur substantial
overhead, like a rewrite of the entire file.

In v2, this overhead is removed, and only changes are written.

### Volumes v2 has a few limits in place

While we work out performance trade-offs and listen to user feedback, we have
put some artificial limits in place.

- Files must be less than one 1 TiB.
- At most 32,768 files can be stored in a single directory.
  Directory depth is unbounded, so the total file count is unbounded.
- Traversing the filesystem can be slower in v2 than in v1, due to demand
  loading of the filesystem tree.

### Upgrading v1 Volumes

Currently, there is no automated tool for upgrading v1 Volumes to v2. We are
planning to implement an automated migration path but for now v1 Volumes need
to be manually migrated by creating a new v2 Volume and either copying files
over from the v1 Volume or writing new files.

To reuse the name of an existing v1 Volume for a new v2 Volume, first stop all
apps that are utilizing the v1 Volume before deleting it. If this is not
feasible, e.g. due to wanting to avoid downtime, use a new name for the v2
Volume.

**Warning:** When deleting an existing Volume, any deployed apps or running
functions utilizing that Volume will cease to function, even if a new Volume is
created with the same name. This is because Volumes are identified with opaque
unique IDs that are resolved at application deployment or start time. A newly
created Volume with the same name as a deleted Volume will have a new Volume ID
and any deployed or running apps will still be referring to the old ID until
these apps are re-deployed or restarted.

In order to create a new volume and copy data over from the old volume, you can
use a tool like `cp` if you intend to copy all the data in one go, or `rsync`
if you want to incrementally copy the data across a longer time span:

```shell
$ modal volume create --version=2 2files2furious
$ modal shell --volume files-and-furious --volume 2files2furious
Welcome to Modal's debug shell!
We've provided a number of utilities for you, like `curl` and `ps`.
# Option 1: use `cp`
root / → cp -rp /mnt/files-and-furious/. /mnt/2files2furious/.
root / → sync /mnt/2files2furious # Ensure changes are persisted before exiting

# Option 2: use `rsync`
root / → apt install -y rsync
root / → rsync -a /mnt/files-and-furious/. /mnt/2files2furious/.
root / → sync /mnt/2files2furious # Ensure changes are persisted before exiting
```

## Further examples

- [Character LoRA fine-tuning](https://modal.com/docs/examples/diffusers_lora_finetune) with model storage on a Volume
- [Protein folding](https://modal.com/docs/examples/chai1) with model weights and output files stored on Volumes
- [Dataset visualization with Datasette](https://modal.com/docs/example/cron_datasette) using a SQLite database on a Volume

#### Storing model weights

# Storing model weights on Modal

Efficiently managing the weights of large models is crucial for optimizing the
build times and startup latency of many ML and AI applications.

Our recommended method for working with model weights is to store them in a Modal [Volume](https://modal.com/docs/guide/volumes),
which acts as a distributed file system, a "shared disk" all of your Modal Functions can access.

## Storing weights in a Modal Volume

To store your model weights in a Volume, you need to either
make the Volume available to a Modal Function that saves the model weights
or upload the model weights into the Volume from a client.

### Saving model weights into a Modal Volume from a Modal Function

If you're already generating the weights on Modal, you just need to
attach the Volume to your Modal Function, making it available for reading and writing:

```python
from pathlib import Path

volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
MODEL_DIR = Path("/models")

@app.function(gpu="any", volumes={MODEL_DIR: volume})  # attach the Volume
def train_model(data, config):
    import run_training

    model = run_training(config, data)
    model.save(config, MODEL_DIR)
```

Volumes are attached by including them in a dictionary that maps
a path on the remote machine to a `modal.Volume` object.
They look just like a normal file system, so model weights can be saved to them
without adding any special code.

If the model weights are generated outside of Modal and made available
over the Internet, for example by an open-weights model provider
or your own training job on a dedicated cluster,
you can also download them into a Volume from a Modal Function:

```python continuation
@app.function(volumes={MODEL_DIR: volume})
def download_model(model_id):
    import model_hub

    model_hub.download(model_id, local_dir=MODEL_DIR / model_id)
```

Add [Modal Secrets](https://modal.com/docs/guide/secrets) to access weights that require authentication.

See [below](#storing-weights-from-the-hugging-face-hub-on-modal) for
more on downloading from the popular Hugging Face Hub.

### Uploading model weights into a Modal Volume

Instead of pulling weights into a Modal Volume from inside a Modal Function,
you might wish to push weights into Modal from a client,
like your laptop or a dedicated training cluster.

For that, you can use the `batch_upload` method of
[`modal.Volume`](https://modal.com/docs/reference/modal.Volume)s
via the Modal Python client library:

```python continuation
volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)

@app.local_entrypoint()
def main(local_path: str, remote_path: str):
    with volume.batch_upload() as upload:
        upload.put_directory(local_path, remote_path)
```

Alternatively, you can upload model weights using the
[`modal volume`](https://modal.com/docs/reference/cli/volume) CLI command:

```bash
modal volume put model-weights-vol path/to/model path/on/volume
```

### Mounting cloud buckets as Modal Volumes

If your model weights are already in cloud storage,
for example in an S3 bucket, you can connect them
to Modal Functions with a `CloudBucketMount`.

See [the guide](https://modal.com/docs/guide/cloud-bucket-mounts) for details.

## Reading model weights from a Modal Volume

You can read weights from a Volume as you would normally read them
from disk, so long as you attach the Volume to your Function.

```python continuation
@app.function(gpu="any", volumes={MODEL_DIR: volume})
def inference(prompt, model_id):
    import load_model

    model = load_model(MODEL_DIR / model_id)
    model.run(prompt)
```

## Storing weights in the Modal Image

It is also possible to store weights in your Function's Modal [Image](https://modal.com/docs/guide/images),
the private file system state that a Function sees when it starts up.
The weights might be downloaded via shell commands with [`Image.run_commands`](https://modal.com/docs/guide/images)
or downloaded using a Python function with [`Image.run_function`](https://modal.com/docs/guide/images).

We recommend storing model weights in a Modal [Volume](https://modal.com/docs/guide/volumes),
as described [above](#storing-weights-in-a-modal-volume). Performance is similar
for the two methods. Volumes are more flexible.
Images are rebuilt when their definition changes, starting from the changed layer,
which increases reproducibility for some builds but leads to unnecessary extra downloads
in most cases.

## Optimizing model weight reads with `@enter`

In the above code samples, weights are loaded from disk into memory each time
the `inference` function is run. This isn't so bad if inference is much
slower than model loading (e.g. it is run on very large datasets)
or if the model loading logic is smart enough to skip reloading.

To guarantee a particular model's weights are only loaded once, you can use the `@enter`
[container lifecycle hook](https://modal.com/docs/guide/lifecycle-functions)
to load the weights only when a new container starts.

```python continuation
MODEL_ID = "some-model-id"

@app.cls(gpu="any", volumes={MODEL_DIR: volume})
class Model:
    @modal.enter()
    def setup(self, model_id=MODEL_ID):
        import load_model

        self.model = load_model(MODEL_DIR, model_id)

    @modal.method()
    def inference(self, prompt):
        return self.model.run(prompt)
```

Note that methods decorated with `@enter` can't be passed dynamic arguments.

If you need to load a single but possibly different model on each container start, you can
[parametrize](https://modal.com/docs/guide/parametrized-functions) your Modal Cls.
Below, we use the `modal.parameter` syntax.

```python continuation
@app.cls(gpu="any", volumes={MODEL_DIR: volume})
class ParametrizedModel:
    model_id: str = modal.parameter()

    @modal.enter()
    def setup(self):
        import load_model

        self.model = load_model(MODEL_DIR, self.model_id)

    @modal.method()
    def inference(self, prompt):
        return self.model.run(prompt)
```

## Storing weights from the Hugging Face Hub on Modal

The [Hugging Face Hub](https://huggingface.co/models) has over 1,000,000 models
with weights available for download.

The snippet below shows some additional tricks for downloading models
from the Hugging Face Hub on Modal.

```python
from typing import Optional
from pathlib import Path

import modal

# create a Volume, or retrieve it if it exists
volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
MODEL_DIR = Path("/models")

# define dependencies for downloading model
download_image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub[hf_transfer]")  # install fast Rust download client
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # and enable it
)
app = modal.App()

@app.function(
    volumes={MODEL_DIR.as_posix(): volume},  # "mount" the Volume, sharing it with your function
    image=download_image,  # only download dependencies needed here
)
def download_model(
    repo_id: str = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    revision: Optional[str] = None,  # include a revision to prevent surprises!
):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, local_dir=MODEL_DIR / repo_id, revision=revision)
    print(f"Model downloaded to {MODEL_DIR / repo_id}")
```

#### Cloud bucket mounts

# Cloud bucket mounts

The [`modal.CloudBucketMount`](https://modal.com/docs/reference/modal.CloudBucketMount) is a
mutable volume that allows for both reading and writing files from a cloud
bucket. It supports AWS S3, Cloudflare R2, and Google Cloud Storage buckets.

Cloud bucket mounts are built on top of AWS'
[`mountpoint`](https://github.com/awslabs/mountpoint-s3) technology and inherits
its limitations. See the [Limitations and troubleshooting](#limitations-and-troubleshooting) section for more details.

## Mounting Cloudflare R2 buckets

`CloudBucketMount` enables Cloudflare R2 buckets to be mounted as file system
volumes. Because Cloudflare R2 is
[S3-Compatible](https://developers.cloudflare.com/r2/api/s3/api/) the setup is
very similar between R2 and S3. See
[modal.CloudBucketMount](https://modal.com/docs/reference/modal.CloudBucketMount#modalcloudbucketmount)
for usage instructions.

When creating the R2 API token for use with the mount, you need to have the
ability to read, write, and list objects in the specific buckets you will mount.
You do _not_ need admin permissions, and you should _not_ use "Client IP Address
Filtering".

## Mounting Google Cloud Storage buckets

`CloudBucketMount` enables Google Cloud Storage (GCS) buckets to be mounted as file system
volumes. See [modal.CloudBucketMount](https://modal.com/docs/reference/modal.CloudBucketMount#modalcloudbucketmount)
for GCS setup instructions.

## Mounting S3 buckets

`CloudBucketMount` enables S3 buckets to be mounted as file system volumes. To
interact with a bucket, you must have the appropriate IAM permissions configured
(refer to the section on [IAM Permissions](#iam-permissions)).

```python
import modal
import subprocess

app = modal.App()

s3_bucket_name = "s3-bucket-name"  # Bucket name not ARN.
s3_access_credentials = modal.Secret.from_dict({
    "AWS_ACCESS_KEY_ID": "...",
    "AWS_SECRET_ACCESS_KEY": "...",
    "AWS_REGION": "..."
})

@app.function(
    volumes={
        "/my-mount": modal.CloudBucketMount(s3_bucket_name, secret=s3_access_credentials)
    }
)
def f():
    subprocess.run(["ls", "/my-mount"])
```

### Specifying S3 bucket region

Amazon S3 buckets are associated with a single AWS Region. [`Mountpoint`](https://github.com/awslabs/mountpoint-s3) attempts to automatically detect the region for your S3 bucket at startup time and directs all S3 requests to that region. However, in certain scenarios, like if your container is running on an AWS worker in a certain region, while your bucket is in a different region, this automatic detection may fail.

To avoid this issue, you can specify the region of your S3 bucket by adding an `AWS_REGION` key to your Modal secrets, as in the code example above.

### Using AWS temporary security credentials

`CloudBucketMount`s also support AWS temporary security credentials by passing
the additional environment variable `AWS_SESSION_TOKEN`. Temporary credentials
will expire and will not get renewed automatically. You will need to update
the corresponding Modal Secret in order to prevent failures.

You can get temporary credentials with the [AWS CLI](https://aws.amazon.com/cli/) with:

```shell
$ aws configure export-credentials --format env
export AWS_ACCESS_KEY_ID=XXX
export AWS_SECRET_ACCESS_KEY=XXX
export AWS_SESSION_TOKEN=XXX...
```

All these values are required.

### Using OIDC identity tokens

Modal provides [OIDC integration](https://modal.com/docs/guide/oidc-integration) and will automatically generate identity tokens to authenticate to AWS.
OIDC eliminates the need for manual token passing through Modal secrets and is based on short-lived tokens, which limits the window of exposure if a token is compromised.
To use this feature, you must [configure AWS to trust Modal's OIDC provider](https://modal.com/docs/guide/oidc-integration#step-1-configure-aws-to-trust-modals-oidc-provider)
and [create an IAM role that can be assumed by Modal Functions](https://modal.com/docs/guide/oidc-integration#step-2-create-an-iam-role-that-can-be-assumed-by-modal-functions).

Then, you specify the IAM role that your Modal Function should assume to access the S3 bucket.

```python
import modal

app = modal.App()

s3_bucket_name = "s3-bucket-name"
role_arn = "arn:aws:iam::123456789abcd:role/s3mount-role"

@app.function(
    volumes={
        "/my-mount": modal.CloudBucketMount(
            bucket_name=s3_bucket_name,
            oidc_auth_role_arn=role_arn
        )
    }
)
def f():
    subprocess.run(["ls", "/my-mount"])
```

### Mounting a path within a bucket

To mount only the files under a specific subdirectory, you can specify a path prefix using `key_prefix`.
Since this prefix specifies a directory, it must end in a `/`.
The entire bucket is mounted when no prefix is supplied.

```python
import modal
import subprocess

app = modal.App()

s3_bucket_name = "s3-bucket-name"
prefix = 'path/to/dir/'

s3_access_credentials = modal.Secret.from_dict({
    "AWS_ACCESS_KEY_ID": "...",
    "AWS_SECRET_ACCESS_KEY": "...",
})

@app.function(
    volumes={
        "/my-mount": modal.CloudBucketMount(
            bucket_name=s3_bucket_name,
            key_prefix=prefix,
            secret=s3_access_credentials
        )
    }
)
def f():
    subprocess.run(["ls", "/my-mount"])
```

This will only mount the files in the bucket `s3-bucket-name` that are prefixed by `path/to/dir/`.

### Read-only mode

To mount a bucket in read-only mode, set `read_only=True` as an argument.

```python
import modal
import subprocess

app = modal.App()

s3_bucket_name = "s3-bucket-name"  # Bucket name not ARN.
s3_access_credentials = modal.Secret.from_dict({
    "AWS_ACCESS_KEY_ID": "...",
    "AWS_SECRET_ACCESS_KEY": "...",
})

@app.function(
    volumes={
        "/my-mount": modal.CloudBucketMount(s3_bucket_name, secret=s3_access_credentials, read_only=True)
    }
)
def f():
    subprocess.run(["ls", "/my-mount"])
```

While S3 mounts support both write and read operations, they are optimized for
reading large files sequentially. Certain file operations, such as renaming
files, are not supported. For a comprehensive list of supported operations,
consult the
[Mountpoint documentation](https://github.com/awslabs/mountpoint-s3/blob/main/doc/SEMANTICS.md).

### IAM permissions

To utilize `CloudBucketMount` for reading and writing files from S3 buckets,
your IAM policy must include permissions for `s3:PutObject`,
`s3:AbortMultipartUpload`, and `s3:DeleteObject`. These permissions are not
required for mounts configured with `read_only=True`.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ModalListBucketAccess",
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": ["arn:aws:s3:::"]
    },
    {
      "Sid": "ModalBucketAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:AbortMultipartUpload",
        "s3:DeleteObject"
      ],
      "Resource": ["arn:aws:s3:::/*"]
    }
  ]
}
```

## Limitations and troubleshooting

Cloud Bucket Mounts have certain limitations that do not apply to [Volumes](https://modal.com/docs/guide/volumes).
These limitations are primarily around the way that files can be opened and edited in Cloud Bucket Mounts. For
a comprehensive list of limitations, see the [Mountpoint troubleshooting documentation](https://github.com/awslabs/mountpoint-s3/blob/a6179c72bfc237a1fdd06eb4a0863ca537f8d8a7/doc/TROUBLESHOOTING.md)
and the [Mountpoint semantics documentation](https://github.com/awslabs/mountpoint-s3/blob/main/doc/SEMANTICS.md).

The most common issues that users encounter are:

- Files cannot be opened in append mode.
- Files cannot be written to at arbitrary offsets i.e. `seek` and write are not supported together.
- To write to a file, you must open it in `truncate` mode.

These operations typically result in a `PermissionError: [Errno 1] Operation not permitted` error.

If you need these features, give [Volumes](https://modal.com/docs/guide/volumes) a try! If you need these features in S3
and are willing to pay extra for your bucket, you may be able to use [S3 Express](https://aws.amazon.com/s3/storage-classes/express-one-zone/).
Contact us [in our Slack](https://modal.com/slack) if you're interested in using S3 Express.

### Writing files in append mode

If you're using a library which must open a file in append mode, it's best to write to a temporary file
and then move it to your bucket's mount path. A similar approach can be used to write to a file at an arbitrary offset.

```python notest
import tempfile
import shutil

@app.function(
    volumes={"/bucket": modal.CloudBucketMount("my-bucket", secret=s3_credentials)}
)
def append_to_log():
    # Write to a temporary file that supports append mode
    with tempfile.NamedTemporaryFile(mode='a', delete=False) as temp_file:
        temp_file.write("Log entry 1\n")
        temp_file.write("Log entry 2\n")
        temp_path = temp_file.name

    # Move the completed file to the bucket mount
    shutil.move(temp_path, "/bucket/logfile.txt")
```

### Creating a file without a parent directory

If you try to create a file in a directory that doesn't exist, you'll get a `Operation not permitted` error.
To fix this, create the parent directory first with `Path(dst).parent.mkdir(exist_ok=True, parents=True)`.

### Using `np.savez`

`np.savez` seeks to random offsets in a file, making it unsafe for Cloud Bucket Mounts. If your file is large,
you can write it to a temporary file and then move it to your bucket's mount path. If it's small, however,
you can solve this with an in-memory buffer:

```python notest
import io
import numpy as np
import shutil

data = np.random.rand(1000, 512)

# 1. Build the archive entirely in memory
tmp = io.BytesIO()
np.savez_compressed(tmp, array=data)

# 2. Copy it once, sequentially, to the mount point
dest = "/bucket/data.npz"
with open(dest, "wb") as f:
    shutil.copyfileobj(tmp, f)
```

### Torchtune writing checkpoint files

Old versions of [Torchtune](https://github.com/pytorch/torchtune) are incompatible with Cloud Bucket Mounts.
Upgrade to a version greater than or equal to `0.6.1` to ensure checkpoints can be written to the bucket.

### Using the TensorBoard `SummaryWriter`

The TensorBoard `SummaryWriter` opens log files in append mode. These files are quite small, though,
so we recommend writing to a temporary directory and using the [Watchdog](https://github.com/gorakhargosh/watchdog)
Python library to copy the files to the bucket mount path as they come in.

This is a case where it may be worth it to use [Volumes](https://modal.com/docs/guide/volumes) instead - in particular,
training logs are sometimes not subject to the same compliance requirements that force something like checkpoints
or model weights to be stored in a secure location. We even have an example of
[how to use TensorBoard on Volumes](https://modal.com/docs/examples/torch_profiling#serving-tensorboard-on-modal-to-view-pytorch-profiles-and-traces).

#### Dicts

# Dicts

Modal Dicts provide distributed key-value storage to your Modal Apps.

```python runner:ModalRunner
import modal

app = modal.App()
kv = modal.Dict.from_name("kv", create_if_missing=True)

@app.local_entrypoint()
def main(key="cloud", value="dictionary", put=True):
    if put:
        kv[key] = value
    print(f"{key}: {kv[key]}")
```

This page is a high-level guide to using Modal Dicts.
For reference documentation on the `modal.Dict` object, see
[this page](https://modal.com/docs/reference/modal.Dict).
For reference documentation on the `modal dict` CLI command, see
[this page](https://modal.com/docs/reference/cli/dict).

## Modal Dicts are Python dicts in the cloud

Dicts provide distributed key-value storage to your Modal Apps.
Much like a standard Python dictionary, a Dict lets you store and retrieve
values using keys. However, unlike a regular dictionary, a Dict in Modal is
accessible from anywhere, concurrently and in parallel.

```python
# create a remote Dict
dictionary = modal.Dict.from_name("my-dict", create_if_missing=True)

dictionary["key"] = "value"  # set a value from anywhere
value = dictionary["key"]    # get a value from anywhere
```

Dicts are persisted, which means that the data in the dictionary is
stored and can be retrieved even after the application is redeployed.

## You can access Modal Dicts asynchronously

Modal Dicts live in the cloud, which means reads and writes
against them go over the network. That has some unavoidable latency overhead,
relative to just reading from memory, of a few dozen ms.
Reads from Dicts via `["key"]`-style indexing are synchronous,
which means that latency is often directly felt by the application.

But like all Modal objects, you can also interact with Dicts asynchronously
by putting the `.aio` suffix on methods -- in this case, `put` and `get`,
which are synonyms for bracket-based indexing.
Just add the `async` keyword to your `local_entrypoint`s or remote Functions
and `await` the method calls.

```python runner:ModalRunner
import modal

app = modal.App()
dictionary = modal.Dict.from_name("async-dict", create_if_missing=True)

@app.local_entrypoint()
async def main():
    await dictionary.put.aio("key", "value")  # setting a value asynchronously
    assert await dictionary.get.aio("key")   # getting a value asyncrhonrously
```

See the guide to [asynchronous functions](https://modal.com/docs/guide/async) for more
information.

## Modal Dicts are not _exactly_ Python dicts

Python dicts can have keys of any hashable type and values of any type.

You can store Python objects of any serializable type within Dicts as keys or values.

Objects are serialized using [`cloudpickle`](https://github.com/cloudpipe/cloudpickle),
so precise support is inherited from that library. `cloudpickle` can serialize a surprising variety of objects,
like `lambda` functions or even Python modules, but it can't serialize a few things that don't
really make sense to serialize, like live system resources (sockets, writable file descriptors).

Note that you will need to have the library defining the type installed in the environment
where you retrieve the object so that it can be deserialized.

```python runner:ModalRunner
import modal

app = modal.App()
dictionary = modal.Dict.from_name("funky-dict", create_if_missing=True)

@app.function(image=modal.Image.debian_slim().pip_install("numpy"))
def fill():
    import numpy

    dictionary["numpy"] = numpy
    dictionary["modal"] = modal
    dictionary[dictionary] = dictionary  # don't try this at home!

@app.local_entrypoint()
def main():
    fill.remote()
    print(dictionary["modal"])
    print(dictionary[dictionary]["modal"].Dict)
    # print(dictionary["numpy"])  # DeserializationError, if no numpy locally
```

Unlike with normal Python dictionaries, updates to mutable value types will not
be reflected in other containers unless the updated object is explicitly put
back into the Dict. As a consequence, patterns like chained updates
(`my_dict["outer_key"]["inner_key"] = value`) cannot be used the same way as
they would with a local dictionary.

Currently, the per-object size limit is 100 MiB and the maximum number of entries
per update is 10,000. It's recommended to use Dicts for smaller objects (under 5 MiB).
Each object in the Dict will expire after 7 days of inactivity (no reads or writes).

Dicts also provide a locking primitive. See
[this blog post](https://modal.com/blog/cache-dict-launch) for details.

#### Queues

# Queues

Modal Queues provide distributed FIFO queues to your Modal Apps.

```python runner:ModalRunner
import modal

app = modal.App()
queue = modal.Queue.from_name("simple-queue", create_if_missing=True)

def producer(x):
    queue.put(x)  # adding a value

@app.function()
def consumer():
    return queue.get()  # retrieving a value

@app.local_entrypoint()
def main(x="some object"):
    # produce and consume tasks from local or remote code
    producer(x)
    print(consumer.remote())
```

This page is a high-level guide to using Modal Queues.
For reference documentation on the `modal.Queue` object, see
[this page](https://modal.com/docs/reference/modal.Queue).
For reference documentation on the `modal queue` CLI command, see
[this page](https://modal.com/docs/reference/cli/queue).

## Modal Queues are Python queues in the cloud

Like [Python `Queue`s](https://docs.python.org/3/library/queue.html),
Modal Queues are multi-producer, multi-consumer first-in-first-out (FIFO) queues.

Queues are particularly useful when you want to handle tasks or process
data asynchronously, or when you need to pass messages between different
components of your distributed system.

Queues are cleared 24 hours after the last `put` operation and are backed by
a replicated in-memory database, so persistence is likely, but not guaranteed.
As such, `Queue`s are best used for communication between active functions and
not relied on for persistent storage.

[Please get in touch](mailto:support@modal.com) if you need durability for Queue objects.

## Queues are partitioned by key

Queues are split into separate FIFO partitions via a string key. By default, one
partition (corresponding to an empty key) is used.

A single `Queue` can contain up to 100,000 partitions, each with up to 5,000
items. Each item can be up to 1 MiB. These limits also apply to the default
partition.

Each partition has an independent TTL, by default 24 hours.
Lower TTLs can be specified by the `partition_ttl` argument in the `put` or
`put_many` methods.

```python runner:ModalRunner
import modal

app = modal.App()
my_queue = modal.Queue.from_name("partitioned-queue", create_if_missing=True)

@app.local_entrypoint()
def main():
    # clear all elements, start from a clean slate
    my_queue.clear()

    my_queue.put("some value")  # first in
    my_queue.put(123)

    assert my_queue.get() == "some value"  # first out
    assert my_queue.get() == 123

    my_queue.put(0)
    my_queue.put(1, partition="foo")
    my_queue.put(2, partition="bar")

    # Default and "foo" partition are ignored by the get operation.
    assert my_queue.get(partition="bar") == 2

    # Set custom 10s expiration time on "foo" partition.
    my_queue.put(3, partition="foo", partition_ttl=10)

    # (beta feature) Iterate through items in place (read immutably)
    my_queue.put(1)
    assert [v for v in my_queue.iterate()] == [0, 1]
```

## You can access Modal Queues synchronously or asynchronously, blocking or non-blocking

Queues are synchronous and blocking by default. Consumers will block and wait
on an empty Queue and producers will block and wait on a full Queue,
both with an `Optional`, configurable `timeout`. If the `timeout` is `None`,
they will wait indefinitely. If a `timeout` is provided, `get` methods will raise
[`queue.Empty`](https://docs.python.org/3/library/queue.html#queue.Empty)
exceptions and `put` methods will raise
[`queue.Full`](https://docs.python.org/3/library/queue.html#queue.Full)
exceptions, both from the Python standard library.

The `get` and `put` methods can be made non-blocking by setting the `block` argument to `False`.
They raise `queue` exceptions without waiting on the `timeout`.

Queues are stored in the cloud, so all interactions require communication over the network.
This adds some extra latency to calls, apart from the `timeout`, on the order of tens of milliseconds.
To avoid this latency impacting application latency, you can asynchronously interact with Queues
by adding the `.aio` function suffix to access methods.

```python notest
@app.local_entrypoint()
async def main(value=None):
    await my_queue.put.aio(value or 200)
    assert await my_queue.get.aio() == value
```

See the guide to [asynchronous functions](https://modal.com/docs/guide/async) for more
information.

## Modal Queues are not _exactly_ Python Queues

Python Queues can have values of any type.

Modal Queues can store Python objects of any serializable type.

Objects are serialized using [`cloudpickle`](https://github.com/cloudpipe/cloudpickle),
so precise support is inherited from that library. `cloudpickle` can serialize a surprising variety of objects,
like `lambda` functions or even Python modules, but it can't serialize a few things that don't
really make sense to serialize, like live system resources (sockets, writable file descriptors).

Note that you will need to have the library defining the type installed in the environment
where you retrieve the object so that it can be deserialized.

```python runner:ModalRunner
import modal

app = modal.App()
queue = modal.Queue.from_name("funky-queue", create_if_missing=True)
queue.clear()  # start from a clean slate

@app.function(image=modal.Image.debian_slim().pip_install("numpy"))
def fill():
    import numpy

    queue.put(modal)
    queue.put(queue)  # don't try this at home!
    queue.put(numpy)

@app.local_entrypoint()
def main():
    fill.remote()
    print(queue.get().Queue)
    print(queue.get())
    # print(queue.get())  # DeserializationError, if no torch locally
```

#### Dataset ingestion

# Large dataset ingestion

This guide provides best practices for downloading, transforming, and storing large datasets within
Modal. A dataset is considered large if it contains hundreds of thousands of files and/or is over
100 GiB in size.

These guidelines ensure that large datasets can be ingested fully and reliably.

## Configure your Function for heavy disk usage

Large datasets should be downloaded and transformed using a `modal.Function` and stored
into a `modal.CloudBucketMount`. We recommend backing the latter with a Cloudflare R2 bucket,
because Cloudflare does not charge network egress fees and has lower GiB/month storage costs than AWS S3.

This `modal.Function` should specify a large `timeout` because large dataset processing can take hours,
and it should request a larger ephemeral disk in cases where the dataset being downloaded and processed
is hundreds of GiBs.

```python
@app.function(
    volumes={
        "/mnt": modal.CloudBucketMount(
            "datasets",
            bucket_endpoint_url="https://abc123example.r2.cloudflarestorage.com",
            secret=modal.Secret.from_name("cloudflare-r2-datasets"),
        )
    },
    ephemeral_disk=1000 * 1000,  # 1 TiB
    timeout=60 * 60 * 12,  # 12 hours

)
def download_and_transform() -> None:
    ...
```

### Use compressed archives on Modal Volumes

`modal.Volume`s are designed for storing tens of thousands of individual files,
but not for hundreds of thousands or millions of files.
However they can be still be used for storing large datasets if files are first combined and compressed
in a dataset transformation step before saving them into the Volume.

See the [transforming](#transforming) section below for more details.

## Experimentation

Downloading and transforming large datasets can be fiddly. While iterating on a reliable ingestion program
it is recommended to start a long-running `modal.Function` serving a JupyterHub server so that you can maintain
disk state in the face of application errors.

See the [running Jupyter server within a Modal function](https://github.com/modal-labs/modal-examples/blob/main/11_notebooks/jupyter_inside_modal.py) example as base code.

## Downloading

The raw dataset data should be first downloaded into the container at `/tmp/` and not placed
directly into the mounted volume. This serves a couple purposes.

1. Certain download libraries and tools (e.g. `wget`) perform filesystem operations not supported properly by `CloudBucketMount`.
2. The raw dataset data may need to be transformed before use, in which case it is wasteful to store it permanently.

This snippet shows the basic download-and-copy procedure:

```python notest
import pathlib
import shutil
import subprocess

tmp_path = pathlib.Path("/tmp/imagenet/")
vol_path = pathlib.Path("/mnt/imagenet/")
filename = "imagenet-object-localization-challenge.zip"
# 1. Download into /tmp/
subprocess.run(
    f"kaggle competitions download -c imagenet-object-localization-challenge --path {tmp_path}",
    shell=True,
    check=True
)
vol_path.mkdir(exist_ok=True)
# 2. Copy (without transform) into mounted volume.
shutil.copyfile(tmp_path / filename, vol_path / filename)
```

## Transforming

When ingesting a large dataset it is sometimes necessary to transform it before storage, so that it is in
an optimal format for loading at runtime. A common kind of necessary transform is gzip decompression. Very large
datasets are often gzipped for storage and network transmission efficiency, but gzip decompression (80 MiB/s)
is hundreds of times slower than reading from a solid state drive (SSD)
and should be done once before storage to avoid decompressing on every read against the dataset.

Transformations should be performed after storing the raw dataset in `/tmp/`. Performing transformations almost always increases container disk usage and this is where the [`ephemeral_disk` parameter](https://modal.com/docs/reference/modal.App#function) parameter becomes important. For example, a
100 GiB raw, compressed dataset may decompress to into 500 GiB, occupying 600 GiB of container disk space.

Transformations should also typically be performed against `/tmp/`. This is because

1. transforms can be IO intensive and IO latency is lower against local SSD.
2. transforms can create temporary data which is wasteful to store permanently.

## Examples

The best practices offered in this guide are demonstrated in the [`modal-examples` repository](https://github.com/modal-labs/modal-examples/tree/main/12_datasets).

The examples include these popular large datasets:

- [ImageNet](https://www.image-net.org/), the image labeling dataset that kicked off the deep learning revolution
- [COCO](https://cocodataset.org/#download), the Common Objects in COntext dataset of densely-labeled images
- [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/), the Stable Diffusion training dataset
- Data derived from the [Big "Fantastic" Database](https://bfd.mmseqs.com/),
  [Protein Data Bank](https://www.wwpdb.org/), and [UniProt Database](https://www.uniprot.org/)
  used in training the [RoseTTAFold](https://github.com/RosettaCommons/RoseTTAFold) protein structure model

### Modal Sandboxes

#### Sandboxes

# Sandboxes

In addition to the Function interface, Modal has a direct
interface for defining containers _at runtime_ and securely running arbitrary code
inside them.

This can be useful if, for example, you want to:

- Execute code generated by a language model.
- Create isolated environments for running untrusted code.
- Check out a git repository and run a command against it, like a test suite, or
  `npm lint`.
- Run containers with arbitrary dependencies and setup scripts.

Each individual job is called a **Sandbox** and can be created using the
[`Sandbox.create`](https://modal.com/docs/reference/modal.Sandbox#create) constructor:

```python notest
import modal

app = modal.App.lookup("my-app", create_if_missing=True)

sb = modal.Sandbox.create(app=app)

p = sb.exec("python", "-c", "print('hello')", timeout=3)
print(p.stdout.read())

p = sb.exec("bash", "-c", "for i in {1..10}; do date +%T; sleep 0.5; done", timeout=5)
for line in p.stdout:
    # Avoid double newlines by using end="".
    print(line, end="")

sb.terminate()
```

**Note:** you can run the above example as a script directly with `python my_script.py`. `modal run` is not needed here since there is no [entrypoint](https://modal.com/docs/guide/apps#entrypoints-for-ephemeral-apps).

Sandboxes require an [`App`](https://modal.com/docs/guide/apps) to be passed when spawned from outside
of a Modal container. You may pass in a regular `App` object or look one up by name with
[`App.lookup`](https://modal.com/docs/reference/modal.App#lookup). The `create_if_missing` flag on `App.lookup`
will create an `App` with the given name if it doesn't exist.

## Lifecycle

### Timeouts

Sandboxes have a default maximum lifetime of 5 minutes. You can change this by passing
a `timeout` of up to 24 hours to the `Sandbox.create(...)` function.

```python notest
sb = modal.Sandbox.create(app=my_app, timeout=600) # 10 minutes
```

If you need a Sandbox to run for more than 24 hours, we recommend using
[Filesystem Snapshots](https://modal.com/docs/guide/sandbox-snapshots) to preserve its state,
and then restore from that snapshot with a subsequent Sandbox.

### Idle Timeouts

Sandboxes can also be automatically terminated after a period of inactivity - you can do this by setting the `idle_timeout` parameter. A Sandbox is considered active if any of the following are true:

1. It has an active [command](https://modal.com/docs/guide/sandbox-spawn) running (via [`sb.exec(...)`](https://modal.com/docs/reference/modal.Sandbox#exec))
2. Its stdin is being written to (via [`sb.stdin.write()`](https://modal.com/docs/reference/modal.Sandbox#stdin))
3. It has an open TCP connection over one of its [Tunnels](https://modal.com/docs/guide/tunnels)

## Configuration

Sandboxes support nearly all configuration options found in regular `modal.Function`s.
Refer to [`Sandbox.create`](https://modal.com/docs/reference/modal.Sandbox#create) for further documentation
on Sandbox configs.

For example, Images and Volumes can be used just as with functions:

```python notest
sb = modal.Sandbox.create(
    image=modal.Image.debian_slim().pip_install("pandas"),
    volumes={"/data": modal.Volume.from_name("my-volume")},
    workdir="/repo",
    app=my_app,
)
```

## Environments

### Environment variables

You can set environment variables using inline secrets:

```python notest
secret = modal.Secret.from_dict({"MY_SECRET": "hello"})

sb = modal.Sandbox.create(
    secrets=[secret],
    app=my_app,
)
p = sb.exec("bash", "-c", "echo $MY_SECRET")
print(p.stdout.read())
```

### Custom Images

Sandboxes support custom images just as Functions do. However, while you'll typically
invoke a Modal Function with the `modal run` cli, you typically spawn a Sandbox
with a simple `python` call. As such, you need to manually enable output streaming
to see your image build logs:

```python notest
image = modal.Image.debian_slim().pip_install("pandas", "numpy")

with modal.enable_output():
    sb = modal.Sandbox.create(image=image, app=my_app)
```

### Dynamically defined environments

Note that any valid `Image` or `Mount` can be used with a Sandbox, even if those
images or mounts have not previously been defined. This also means that Images and
Mounts can be built from requirements at **runtime**. For example, you could
use a language model to write some code and define your image, and then spawn a
Sandbox with it. Check out [devlooper](https://github.com/modal-labs/devlooper)
for a concrete example of this.

## Running a Sandbox with an entrypoint

In most cases, Sandboxes are treated as a generic container that can run arbitrary
commands. However, in some cases, you may want to run a single command or script
as the entrypoint of the Sandbox. You can do this by passing string arguments to the
Sandbox constructor:

```python notest
sb = modal.Sandbox.create("python", "-m", "http.server", "8080", app=my_app, timeout=10)
for line in sb.stdout:
    print(line, end="")
```

This functionality is most useful for running long-lived services that you want
to keep running in the background. See our [Jupyter notebook example](https://modal.com/docs/examples/jupyter_sandbox)
for a more concrete example of this.

## Referencing Sandboxes from other code

If you have a running Sandbox, you can retrieve it using the [`Sandbox.from_id`](https://modal.com/docs/reference/modal.Sandbox#from_id)
method.

```python notest
sb = modal.Sandbox.create(app=my_app)
sb_id = sb.object_id

# ... later in the program ...

sb2 = modal.Sandbox.from_id(sb_id)

p = sb2.exec("echo", "hello")
print(p.stdout.read())
sb2.terminate()
```

A common use case for this is keeping a pool of Sandboxes available for executing tasks
as they come in. You can keep a list of `object_id`s of Sandboxes that are "open" and
reuse them, closing over the `object_id` in whatever function is using them.

## Logging

You can see Sandbox execution logs using `verbose=True`. For example:

```python notest
sb = modal.Sandbox.create(app=my_app, verbose=True)

p = sb.exec("python", "-c", "print('hello')")
print(p.stdout.read())

with sb.open("test.txt", "w") as f:
    f.write("Hello World\n")
```

shows Sandbox logs:

```
Sandbox exec started: python -c print('hello')
Opened file 'test.txt': fd-yErSQzGL9sig6WAjyNgTPR
Wrote to file: fd-yErSQzGL9sig6WAjyNgTPR
Closed file: fd-yErSQzGL9sig6WAjyNgTPR
```

## Named Sandboxes

You can assign a name to a Sandbox when creating it. Each name must be unique within an app -
only one _running_ Sandbox can use a given name at a time. Note that the associated app must be
a deployed app. Once a Sandbox completely stops running, its name becomes available for reuse.
Some applications find Sandbox Names to be useful for ensuring that no more than one Sandbox is
running per resource or project. If a Sandbox with the given name is already running, `create()`
will raise a `modal.exception.AlreadyExistsError`.

```python notest
sb1 = modal.Sandbox.create(app=my_app, name="my-name")
# this will raise a modal.exception.AlreadyExistsError
sb2 = modal.Sandbox.create(app=my_app, name="my-name")
```

A named Sandbox may be fetched from a deployed app using `modal.Sandbox.from_name()` _but only
if the Sandbox is currently running_. If no running Sandbox is found, `from_name()` will raise
a `modal.exception.NotFoundError`.

```python notest
my_app = modal.App.lookup("my-app", create_if_missing=True)
sb1 = modal.Sandbox.create(app=my_app, name="my-name")
# returns the currently running Sandbox with the name "my-name" from the
# deployed app named "my-app".
sb2 = modal.Sandbox.from_name("my-app", "my-name")
assert sb1.object_id == sb2.object_id # sb1 and sb2 refer to the same Sandbox
```

Sandbox Names may contain only alphanumeric characters, dashes, periods, and underscores, and must
be shorter than 64 characters.

## Tagging

Sandboxes can also be tagged with arbitrary key-value pairs. These tags can be used
to filter results in [`Sandbox.list`](https://modal.com/docs/reference/modal.Sandbox#list).

```python notest
sandbox_v1_1 = modal.Sandbox.create("sleep", "10", app=my_app)
sandbox_v1_2 = modal.Sandbox.create("sleep", "20", app=my_app)

sandbox_v1_1.set_tags({"major_version": "1", "minor_version": "1"})
sandbox_v1_2.set_tags({"major_version": "1", "minor_version": "2"})

for sandbox in modal.Sandbox.list(app_id=my_app.app_id):  # All sandboxes.
    print(sandbox.object_id)

for sandbox in modal.Sandbox.list(
    app_id=my_app.app_id,
    tags={"major_version": "1"},
):  # Also all sandboxes.
    print(sandbox.object_id)

for sandbox in modal.Sandbox.list(
    app_id=app.app_id,
    tags={"major_version": "1", "minor_version": "2"},
):  # Just the latest sandbox.
    print(sandbox.object_id)
```

#### Running commands

# Running commands in Sandboxes

Once you have created a Sandbox, you can run commands inside it using the
[`Sandbox.exec`](https://modal.com/docs/reference/modal.Sandbox#exec) method.

```python notest
sb = modal.Sandbox.create(app=my_app)

process = sb.exec("echo", "hello", timeout=3)
print(process.stdout.read())

process = sb.exec("python", "-c", "print(1 + 1)", timeout=3)
print(process.stdout.read())

process = sb.exec(
    "bash",
    "-c",
    "for i in $(seq 1 10); do echo foo $i; sleep 0.1; done",
    timeout=5,
)
for line in process.stdout:
    print(line, end="")

sb.terminate()
```

`Sandbox.exec` returns a [`ContainerProcess`](https://modal.com/docs/reference/modal.container_process#modalcontainer_processcontainerprocess)
object, which allows access to the process's `stdout`, `stderr`, and `stdin`.
The `timeout` parameter ensures that the `exec` command will run for at most
`timeout` seconds.

## Input

The Sandbox and ContainerProcess `stdin` handles are [`StreamWriter`](https://modal.com/docs/reference/modal.io_streams#modalio_streamsstreamwriter)
objects. This object supports flushing writes with both synchronous and asynchronous APIs:

```python notest
import asyncio

sb = modal.Sandbox.create(app=my_app)

p = sb.exec("bash", "-c", "while read line; do echo $line; done")
p.stdin.write(b"foo bar\n")
p.stdin.write_eof()
p.stdin.drain()
p.wait()
sb.terminate()

async def run_async():
    sb = await modal.Sandbox.create.aio(app=my_app)
    p = await sb.exec.aio("bash", "-c", "while read line; do echo $line; done")
    p.stdin.write(b"foo bar\n")
    p.stdin.write_eof()
    await p.stdin.drain.aio()
    await p.wait.aio()
    await sb.terminate.aio()

asyncio.run(run_async())
```

## Output

The Sandbox and ContainerProcess `stdout` and `stderr` handles are [`StreamReader`](https://modal.com/docs/reference/modal.io_streams#modalio_streamsstreamreader)
objects. These objects support reading from the stream in both synchronous and asynchronous manners.
These handles also respect the timeout given to `Sandbox.exec`.

To read from a stream after the underlying process has finished, you can use the `read`
method, which blocks until the process finishes and returns the entire output stream.

```python notest
sb = modal.Sandbox.create(app=my_app)
p = sb.exec("echo", "hello")
print(p.stdout.read())
sb.terminate()
```

To stream output, take advantage of the fact that `stdout` and `stderr` are
iterable:

```python notest
import asyncio

sb = modal.Sandbox.create(app=my_app)

p = sb.exec("bash", "-c", "for i in $(seq 1 10); do echo foo $i; sleep 0.1; done")

for line in p.stdout:
    # Lines preserve the trailing newline character, so use end="" to avoid double newlines.
    print(line, end="")
p.wait()
sb.terminate()

async def run_async():
    sb = await modal.Sandbox.create.aio(app=my_app)
    p = await sb.exec.aio("bash", "-c", "for i in $(seq 1 10); do echo foo $i; sleep 0.1; done")
    async for line in p.stdout:
        # Avoid double newlines by using end="".
        print(line, end="")
    await p.wait.aio()
    await sb.terminate.aio()

asyncio.run(run_async())
```

### Stream types

By default, all streams are buffered in memory, waiting to be consumed by the
client. You can control this behavior with the `stdout` and `stderr` parameters.
These parameters are conceptually similar to the `stdout` and `stderr`
parameters of the [`subprocess`](https://docs.python.org/3/library/subprocess.html#subprocess.DEVNULL) module.

```python notest
from modal.stream_type import StreamType

sb = modal.Sandbox.create(app=my_app)

# Default behavior: buffered in memory.
p = sb.exec(
    "bash",
    "-c",
    "echo foo; echo bar >&2",
    stdout=StreamType.PIPE,
    stderr=StreamType.PIPE,
)
print(p.stdout.read())
print(p.stderr.read())

# Print the stream to STDOUT as it comes in.
p = sb.exec(
    "bash",
    "-c",
    "echo foo; echo bar >&2",
    stdout=StreamType.STDOUT,
    stderr=StreamType.STDOUT,
)
p.wait()

# Discard all output.
p = sb.exec(
    "bash",
    "-c",
    "echo foo; echo bar >&2",
    stdout=StreamType.DEVNULL,
    stderr=StreamType.DEVNULL,
)
p.wait()

sb.terminate()
```

#### Networking and security

# Networking and security

Sandboxes are built to be secure-by-default, meaning that a default Sandbox has
no ability to accept incoming network connections or access your Modal resources.

## Networking

Since Sandboxes may run untrusted code, they have options to restrict their network access.
To block all network access, set `block_network=True` on [`Sandbox.create`](https://modal.com/docs/reference/modal.Sandbox#create).

For more fine-grained networking control, a Sandbox's outbound network access
can be restricted using the `cidr_allowlist` parameter. This parameter takes a
list of CIDR ranges that the Sandbox is allowed to access, blocking all other
outbound traffic.

### Forwarding ports

Sandboxes can also expose TCP ports to the internet. This is useful if,
for example, you want to connect to a web server running inside a Sandbox.

Use the `encrypted_ports` and `unencrypted_ports` parameters of `Sandbox.create`
to specify which ports to forward. You can then access the public URL of a tunnel
using the [`Sandbox.tunnels`](https://modal.com/docs/reference/modal.Sandbox#tunnels) method:

```python notest
import requests
import time

sb = modal.Sandbox.create(
    "python",
    "-m",
    "http.server",
    "12345",
    encrypted_ports=[12345],
    app=my_app,
)

tunnel = sb.tunnels()[12345]

time.sleep(1)  # Wait for server to start.

print(f"Connecting to {tunnel.url}...")
print(requests.get(tunnel.url, timeout=5).text)
```

It is also possible to create an encrypted port that uses `HTTP/2` rather than `HTTP/1.1` with the `h2_ports` option. This will return
a URL that you can make H2 (HTTP/2 + TLS) requests to. If you want to run an `HTTP/2` server inside a sandbox, this feature may be useful.
Here is an example:

```python notest
import time

port = 4359
sb = modal.Sandbox.create(
    app=my_app,
    image=my_image,
    h2_ports = [port],
)
p = sb.exec("python", "my_http2_server.py")

tunnel = sb.tunnels()[port]
time.sleep(1)
print(f"Tunnel URL: {tunnel.url}")
```

For more details on how tunnels work, see the [tunnels guide](https://modal.com/docs/guide/tunnels).

## Security model

In a typical Modal Function, the Function code can call other Modal APIs allowing
it to spawn containers, create and destroy Volumes, read from Dicts and Queues, etc.
Sandboxes, by contrast, are isolated from the main Modal workspace. They have no API
access, meaning the blast radius of any malicious code is limited to the Sandbox
environment.

Sandboxes are built on top of [gVisor](https://gvisor.dev/), a container runtime
by Google that provides strong isolation properties. gVisor has custom logic to
prevent Sandboxes from making malicious system calls, giving you stronger isolation
than standard [runc](https://github.com/opencontainers/runc) containers.

#### File access

# Filesystem Access

There are multiple options for uploading files to a Sandbox and accessing them
from outside the Sandbox.

## Efficient file syncing

To efficiently upload local files to a Sandbox, you can use the
[`add_local_file`](https://modal.com/docs/reference/modal.Image#add_local_file) and
[`add_local_dir`](https://modal.com/docs/reference/modal.Image#add_local_dir) methods on the
[`Image`](https://modal.com/docs/reference/modal.Image) class:

```python notest
sb = modal.Sandbox.create(
    app=my_app,
    image=modal.Image.debian_slim().add_local_dir(
        local_path="/home/user/my_dir",
        remote_path="/app"
    )
)
p = sb.exec("ls", "/app")
print(p.stdout.read())
p.wait()
```

Alternatively, it's possible to use Modal [Volume](https://modal.com/docs/reference/modal.Volume)s or
[CloudBucketMount](https://modal.com/docs/guide/cloud-bucket-mounts)s. These have the benefit that
files created from inside the Sandbox can easily be accessed outside the
Sandbox.

To efficiently upload files to a Sandbox using a Volume, you can use the
[`batch_upload`](https://modal.com/docs/reference/modal.Volume#batch_upload) method on the
`Volume` class - for instance, using an ephemeral Volume that
will be garbage collected when the App finishes:

```python notest
with modal.Volume.ephemeral() as vol:
    import io
    with vol.batch_upload() as batch:
        batch.put_file("local-path.txt", "/remote-path.txt")
        batch.put_directory("/local/directory/", "/remote/directory")
        batch.put_file(io.BytesIO(b"some data"), "/foobar")

    sb = modal.Sandbox.create(
        volumes={"/cache": vol},
        app=my_app,
    )
    p = sb.exec("cat", "/cache/remote-path.txt")
    print(p.stdout.read())
    p.wait()
    sb.terminate()
```

The caller also can access files created in the Volume from the Sandbox, even after the Sandbox is terminated:

```python notest
with modal.Volume.ephemeral() as vol:
    sb = modal.Sandbox.create(
        volumes={"/cache": vol},
        app=my_app,
    )
    p = sb.exec("bash", "-c", "echo foo > /cache/a.txt")
    p.wait()
    sb.terminate()
    sb.wait(raise_on_termination=False)
    for data in vol.read_file("a.txt"):
        print(data)
```

Alternatively, if you want to persist files between Sandbox invocations (useful
if you're building a stateful code interpreter, for example), you can use create
a persisted `Volume` with a dynamically assigned label:

```python notest
session_id = "example-session-id-123abc"
vol = modal.Volume.from_name(f"vol-{session_id}", create_if_missing=True)
sb = modal.Sandbox.create(
    volumes={"/cache": vol},
    app=my_app,
)
p = sb.exec("bash", "-c", "echo foo > /cache/a.txt")
p.wait()
sb.terminate()
sb.wait(raise_on_termination=False)
for data in vol.read_file("a.txt"):
    print(data)
```

File syncing behavior differs between Volumes and CloudBucketMounts. For
Volumes, files are only synced back to the Volume when the Sandbox terminates.
For CloudBucketMounts, files are synced automatically.

## Filesystem API (Alpha)

If you're less concerned with efficiency of uploads and want a convenient way
to pass data in and out of the Sandbox during execution, you can use our
filesystem API to easily read and write files. The API supports reading
files up to 100 MiB and writes up to 1 GiB in size.

This API is currently in Alpha, and we don't recommend using it for production
workloads.

```python
import modal

app = modal.App.lookup("sandbox-fs-demo", create_if_missing=True)

sb = modal.Sandbox.create(app=app)

with sb.open("test.txt", "w") as f:
    f.write("Hello World\n")

f = sb.open("test.txt", "rb")
print(f.read())
f.close()
```

The filesystem API is similar to Python's built-in [io.FileIO](https://docs.python.org/3/library/io.html#io.FileIO) and supports many of the same methods, including `read`, `readline`, `readlines`, `write`, `flush`, `seek`, and `close`.

We also provide the special methods `replace_bytes` and `delete_bytes`, which may be useful for LLM-generated code.

```python notest
from modal.file_io import delete_bytes, replace_bytes

with sb.open("example.txt", "w") as f:
    f.write("The quick brown fox jumps over the lazy dog")

with sb.open("example.txt", "r+") as f:
    # The quick brown fox jumps over the lazy dog
    print(f.read())

    # The slow brown fox jumps over the lazy dog
    replace_bytes(f, b"slow", start=4, end=9)

    # The slow red fox jumps over the lazy dog
    replace_bytes(f, b"red", start=9, end=14)

    # The slow red fox jumps over the dog
    delete_bytes(f, start=32, end=37)

    f.seek(0)
    print(f.read())

sb.terminate()
```

We additionally provide commands [`mkdir`](https://modal.com/docs/reference/modal.Sandbox#mkdir), [`rm`](https://modal.com/docs/reference/modal.Sandbox#rm), and [`ls`](https://modal.com/docs/reference/modal.Sandbox#ls) to make interacting with the filesystem more ergonomic.

<!-- TODO(WRK-956) -->
<!-- ## File Watching

You can watch files or directories for changes using [`watch`](https://modal.com/docs/reference/modal.Sandbox#watch), which is conceptually similar to [`fsnotify`](https://pkg.go.dev/github.com/fsnotify/fsnotify).

```python notest
from modal.file_io import FileWatchEventType

async def watch(sb: modal.Sandbox):
    event_stream = sb.watch.aio(
        "/watch",
        recursive=True,
        filter=[FileWatchEventType.Create, FileWatchEventType.Modify],
    )
    async for event in event_stream:
        print(event)

async def main():
    app = modal.App.lookup("sandbox-file-watch", create_if_missing=True)
    sb = await modal.Sandbox.create.aio(app=app)
    asyncio.create_task(watch(sb))

    await sb.mkdir.aio("/watch")
    for i in range(10):
        async with await sb.open.aio(f"/watch/bar-{i}.txt", "w") as f:
            await f.write.aio(f"hello-{i}")
``` -->

#### Snapshots

# Snapshots

Sandboxes support snapshotting, allowing you to save your Sandbox's state
and restore it later. This is useful for:

- Creating custom environments for your Sandboxes to run in
- Backing up your Sandbox's state for debugging
- Running large-scale experiments with the same initial state
- Branching your Sandbox's state to test different code changes independently

## Filesystem Snapshots

Filesystem Snapshots are copies of the Sandbox's filesystem at a given point in time.
These Snapshots are [Images](https://modal.com/docs/reference/modal.Image) and can be used to create
new Sandboxes.

To create a Filesystem Snapshot, you can use the
[`Sandbox.snapshot_filesystem()`](https://modal.com/docs/reference/modal.Sandbox#snapshot_filesystem) method:

```python notest
import modal

app = modal.App.lookup("sandbox-fs-snapshot-test", create_if_missing=True)

sb = modal.Sandbox.create(app=app)
p = sb.exec("bash", "-c", "echo 'test' > /test")
p.wait()
assert p.returncode == 0, "failed to write to file"
image = sb.snapshot_filesystem()
sb.terminate()

sb2 = modal.Sandbox.create(image=image, app=app)
p2 = sb2.exec("bash", "-c", "cat /test")
assert p2.stdout.read().strip() == "test"
sb2.terminate()
```

Filesystem Snapshots are optimized for performance: they are calculated as the difference
from your base image, so only modified files are stored. Restoring a Filesystem Snapshot
utilizes the same infrastructure we use to get fast cold starts for your Sandboxes.

Filesystem Snapshots will generally persist indefinitely.

## Memory Snapshots

[Sandboxes memory snapshots](https://modal.com/docs/guide/sandbox-memory-snapshots) are in early preview.
Contact us if this is something you're interested in!

### Modal Notebooks

# Modal Notebooks

Notebooks allow you to write and execute Python code in Modal's cloud, within your browser. It's a hosted Jupyter notebook with:

- Serverless pricing and automatic idle shutdown
- Access to Modal GPUs and compute
- Real-time collaborative editing
- Python Intellisense/LSP support and AI autocomplete
- Support for rich and interactive outputs like images, widgets, and plots

<center>
<video controls autoplay muted playsinline>
<source src="https://modal-cdn.com/Modal-Notebooks-Beta.mp4" type="video/mp4">
</video>
</center>

## Getting started

Open [modal.com/notebooks](https://modal.com/notebooks) in your browser and create a new notebook. You can also upload an `.ipynb` file from your computer.

Once you create a notebook, you can start running cells. Try a simple statement like

```python
print("Hello, Modal!")
```

Or, import a library and create a plot:

```python notest
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-20, 20, 500)
plt.plot(np.cos(x / 3.7 + 0.3), x * np.sin(x))
```

The default notebook image comes with a number of Python packages pre-installed, so you can get started right away. Popular ones include PyTorch, NumPy, Pandas, JAX, Transformers, and Matplotlib. You can find the full image definition [here](https://github.com/modal-labs/modal-client/blob/v1.1.3/modal/experimental/__init__.py#L234-L342). If you need another package, just install it:

```shell
%uv pip install [my-package]
```

All output types work out-of-the-box, including rich HTML, images, [Jupyter Widgets](https://ipywidgets.readthedocs.io/en/latest/), and interactive plots.

## Kernel resources

Just like with Modal Functions, notebooks run in serverless containers. This means you pay only for the CPU cores and memory you use.

If you need more resources, you can change kernel settings in the sidebar. This lets you set the number of CPU cores, memory, and GPU type for your notebook. You can also set a timeout for idle shutdown, which defaults to 10 minutes.

Use any GPU type available in Modal, including up to 8 Nvidia A100s or H100s. You can switch the kernel configuration in seconds!

![Compute profile tab in notebook sidebar](https://modal-cdn.com/cdnbot/compute-profilev9rvmmvw_365a1197.webp)

Note that the CPU and memory settings are _reservations_, so you can usually burst above the request. For example, if you've set the notebook to have 0.5 CPU cores, you'll be billed for that continuously, but you can use up to any available cores on the machine (e.g., 32 CPUs) and will be billed for only the time you use them.

### Notebook pricing

Modal Notebooks are priced simply, by compute usage while the kernel is running. See the [pricing page](https://modal.com/pricing) for rates. Currently the CPU and Memory costs are priced according to Sandboxes. They appear in your [usage dashboard](https://modal.com/settings/usage) under "Sandboxes" as well.

Inactive notebooks do not incur any cost. You are only billed for time the notebook is actively running.

## Custom images, volumes and secrets

Modal Notebooks supports custom images, volumes, and secrets, just like Modal Functions. You can use these to install additional packages, mount persistent storage, or access secrets.

- To use a custom image, you need to have a [deployed Modal Function](https://modal.com/docs/guide/managing-deployments) using that image. Then, search for that function in the sidebar.
- To use a Secret, simply create a [Modal Secret](https://modal.com/secrets) using our wizard and attach it to the notebook, so it can be injected as an environment variable automatically.
- To use a Volume, create a [Modal Volume](https://modal.com/docs/guide/volumes) and attach it to the notebook. This lets you mount high-performance, persistent storage that can be shared across multiple notebooks or functions. They will appear as folders in the `/mnt` directory by default.

### Creating a Custom Image

If you don't have a suitable deployed Modal App already, you can set up your environment to deploy custom images in under a minute using the Modal CLI. First, run `pip install modal`, and define your image in a file like:

```python
import modal

# Image definition here:
image = (
    modal.Image.from_registry("python:3.13-slim")
    .pip_install("requests", "numpy")
    .apt_install("curl", "wget")
    .run_commands(
        "echo 'foo' > /root/hello.txt",
        # ... other commands
    )
)

app = modal.App("notebook-images")

@app.function(image=image)  # You need a Function object to reference the image.
def notebook_image():
    pass
```

Then, make sure you have the Modal CLI (`pip install modal`) and run this command to build and deploy the image:

```bash
modal deploy notebook_images.py
```

For more information on custom images in Modal, see our [guide on defining images](https://modal.com/docs/guide/images).

(Advanced) Note that if you use the [`add_local_file()` or `add_local_dir()` functions](https://modal.com/docs/guide/images#add-local-files-with-add_local_dir-and-add_local_file), you'll need to pass `copy=True` for them to work in Modal Notebooks. This is because they skip creating a custom image and instead mount the files into the function at startup, which won't work in notebooks.

### Creating a Secret

Secrets can be created from the dashboard at [modal.com/secrets](https://modal.com/secrets). We have templates for common credential types, and they are saved as encrypted objects until container startup.

Attacahed secrets become available as environment variables in your notebook.

### Creating a Volume

[Volumes](https://modal.com/docs/guide/volumes) can be created via the files panel on the filesystem tab. This panel can also be used to attach existing Volumes from your Apps or Functions, including those created via the Modal CLI.

Any volumes are attached in the `/mnt` folder in your notebook, and files saved there will be persisted across kernel startups and elsewhere on Modal.

## Access and sharing

Need a colleague—or the whole internet—to see your work? Just click **Share** in the top‑right corner of the notebook editor.

Notebooks are editable by you and teammates in your workspace. To make the notebook view-only to collaborators, the creator of the notebook can change access settings in the "Share" menu. Workspace managers are also allowed to change this setting.

You can also turn on sharing by public, unlisted link. If you toggle this, it allows _anyone with the link_ to open the notebook, even if they are not logged in. Pick **Can view** (default) or **Can view and run** based on your preference. Viewers don’t need a Modal account, so this is perfect for collaborating with stakeholders outside your workspace.

No matter how the notebook is shared, anyone with access can fork and run their own version of it.

## Interactive file viewer

The panel on the left-hand side of the notebook shows a **live view of the container’s filesystem**:

| Feature                 | Details                                                                                                                                                                    |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Browse & preview**    | Click through folders to inspect any file that your code has created or downloaded.                                                                                        |
| **Upload & download**   | Drag-and-drop files from your desktop, or click the **⬆** / **⬇** icons to add new data sets, notebooks, or models—or to save results back to your machine.              |
| **One-click refresh**   | Changes made by your code (for example, writing a CSV) appear instantly; hit the refresh icon if you want to force an update.                                              |
| **Context-aware paths** | The viewer always reflects _exactly_ what your code sees (e.g. `/root`, `/mnt/…`), so you can double-check that that file you just wrote really landed where you expected. |

**Important:** the underlying container is **ephemeral**. Anything stored outside an attached [Volume](https://modal.com/docs/guide/volumes) disappears when the kernel shuts down (after your idle-timeout or when you hit **Stop kernel**). Mount a Volume for data you want to keep across sessions.

The viewer itself is only active while the kernel is running—if the notebook is stopped you’ll see an “empty” state until you start it again.

## Editor features

Modal Notebooks bundle the same productivity tooling you’d expect from a modern IDE.

With Pyright, you get autocomplete, signature help, and on-hover documentation for every installed library.

We also implemented AI-powered code completion using Anthropic's **Claude 4** model. This keeps you in the flow for everything from small snippets to multi-line functions. Just press `Tab` to accept suggestions or `Esc` to dismiss them.

Familiar Jupyter shortcuts (`A`, `B`, `X`, `Y`, `M`, etc.) all work within the notebook, so you can quickly add new cells, delete existing ones, or change cell types.

Finally, we have real-time collaborative editing, so you can work with your team in the same notebook. You can see other users' cursors and edits in real-time, and you can see when others are running cells with you. This makes it easy to pair program or review code together.

## Widgets

Modal Notebooks support [Jupyter Widgets](https://ipywidgets.readthedocs.io/en/latest/), which can be used to create interactive components living in the browser. Currently, Notebooks support all the widgets in the base `ipywidgets` package, except the following:

- Media Widgets (`Audio`, `Video`), try using `IPython.display` outputs instead.
- `Play`
- Controllers (`ControllerAxis`, `ControllerButton`, `Controller`)

Modal Notebooks do not support custom widget packages.

## Cell magic

Modal Notebooks have built-in support for the `%modal` cell magic. This lets you run code in any [deployed Modal Function or Cls](https://modal.com/docs/guide/trigger-deployed-functions), right from your notebook.

For example, if you have previously run `modal deploy` for an app like:

```python notest
import modal

app = modal.App("my-app")

@app.function()
def my_function(s: str):
    return len(s)
```

Then you could access this function from your notebook:

```python notest
%modal from my-app import my_function

my_function.remote("hello, world!")  # returns 13
```

Run `%modal` to see all options. This works for Cls as well, and you can import from different environments or alias them with the `as` keyword.

## Roadmap

The product is in beta, and we're planning to make a lot of improvements over the coming months. Some bigger features on mind:

- **Modal cloud integrations**
  - Expose ports with [Tunnels](https://modal.com/docs/guide/tunnels)
  - Memory snapshots to restore from past notebook sessions
  - Create notebooks from the `modal` CLI
  - Custom image registry
- **Notebook editor**
  - Interactive outline, collapsing sections by headings
  - Reactive cell execution
  - Edit history
  - Integrated debugger (pdb and `%debug`)
- **Documents and sharing**
  - Restore recently deleted notebooks
  - Folders and tags for grouping notebooks
  - Sync with Git repositories

Let us know via [Slack](https://modal.com/slack) if you have any feedback.

### Performance

#### Cold start performance

# Cold start performance

Modal Functions are run in [containers](https://modal.com/docs/guide/images).

If a container is already ready to run your Function, it will be reused.

If not, Modal spins up a new container.
This is known as a _cold start_,
and it is often associated with higher latency.

There are two sources of increased latency during cold starts:

1. inputs may **spend more time waiting** in a queue for a container
   to become ready or "warm".
2. when an input is handled by the container that just started,
   there may be **extra work that only needs to be done on the first invocation**
   ("initialization").

This guide presents techniques and Modal features for reducing the impact of both queueing
and initialization on observed latencies.

If you are invoking Functions with no warm containers
or if you otherwise see inputs spending too much time in the "pending" state,
you should
[target queueing time for optimization](#reduce-time-spent-queueing-for-warm-containers).

If you see some Function invocations taking much longer than others,
and those invocations are the first handled by a new container,
you should
[target initialization for optimization](#reduce-latency-from-initialization).

## Reduce time spent queueing for warm containers

New containers are booted when there are not enough other warm containers to
to handle the current number of inputs.

For example, the first time you send an input to a Function,
there are zero warm containers and there is one input,
so a single container must be booted up.
The total latency for the input will include
the time it takes to boot a container.

If you send another input right after the first one finishes,
there will be one warm container and one pending input,
and no new container will be booted.

Generalizing, there are two factors that affect the time inputs spend queueing:
the time it takes for a container to boot and become warm (which we solve by booting faster)
and the time until a warm container is available to handle an input (which we solve by having more warm containers).

### Warm up containers faster

The time taken for a container to become warm
and ready for inputs can range from seconds to minutes.

Modal's custom container stack has been heavily optimized to reduce this time.
Containers boot in about one second.

But before a container is considered warm and ready to handle inputs,
we need to execute any logic in your code's global scope (such as imports)
or in any
[`modal.enter` methods](https://modal.com/docs/guide/lifecycle-functions).
So if your boots are slow, these are the first places to work on optimization.

For example, you might be downloading a large model from a model server
during the boot process.
You can instead
[download the model ahead of time](https://modal.com/docs/guide/model-weights),
so that it only needs to be downloaded once.

For models in the tens of gigabytes,
this can reduce boot times from minutes to seconds.

### Run more warm containers

It is not always possible to speed up boots sufficiently.
For example, seconds of added latency to load a model may not
be acceptable in an interactive setting.

In this case, the only option is to have more warm containers running.
This increases the chance that an input will be handled by a warm container,
for example one that finishes an input while another container is booting.

Modal currently exposes [three parameters](https://modal.com/docs/guide/scale) that control how
many containers will be warm: `scaledown_window`, `min_containers`,
and `buffer_containers`.

All of these strategies can increase the resources consumed by your Function
and so introduce a trade-off between cold start latencies and cost.

#### Keep containers warm for longer with `scaledown_window`

Modal containers will remain idle for a short period before shutting down. By
default, the maximum idle time is 60 seconds. You can configure this by setting
the `scaledown_window` on the [`@function`](https://modal.com/docs/reference/modal.App#function)
decorator. The value is measured in seconds, and it can be set anywhere between
two seconds and twenty minutes.

```python
import modal

app = modal.App()

@app.function(scaledown_window=300)
def my_idle_greeting():
    return {"hello": "world"}
```

Increasing the `scaledown_window` reduces the chance that subsequent requests
will require a cold start, although you will be billed for any resources used
while the container is idle (e.g., GPU reservation or residual memory
occupancy). Note that containers will not necessarily remain alive for the
entire window, as the autoscaler will scale down more agressively when the
Function is substantially over-provisioned.

#### Overprovision resources with `min_containers` and `buffer_containers`

Keeping already warm containers around longer doesn't help if there are no warm
containers to begin with, as when Functions scale from zero.

To keep some containers warm and running at all times, set the `min_containers`
value on the [`@function`](https://modal.com/docs/reference/modal.App#function) decorator. This
puts a floor on the the number of containers so that the Function doesn't scale
to zero. Modal will still scale up and spin down more containers as the
demand for your Function fluctuates above the `min_containers` value, as usual.

While `min_containers` overprovisions containers while the Function is idle,
`buffer_containers` provisions extra containers while the Function is active.
This "buffer" of extra containers will be idle and ready to handle inputs if
the rate of requests increases. This parameter is particularly useful for
bursty request patterns, where the arrival of one input predicts the arrival of more inputs,
like when a new user or client starts hitting the Function.

```python
import modal

app = modal.App(image=modal.Image.debian_slim().pip_install("fastapi"))

@app.function(min_containers=3, buffer_containers=3)
def my_warm_greeting():
    return "Hello, world!"
```

## Reduce latency from initialization

Some work is done the first time that a function is invoked
but can be used on every subsequent invocation.
This is
[_amortized work_](https://www.cs.cornell.edu/courses/cs312/2006sp/lectures/lec18.html)
done at initialization.

For example, you may be using a large pre-trained model
whose weights need to be loaded from disk to memory the first time it is used.

This results in longer latencies for the first invocation of a warm container,
which shows up in the application as occasional slow calls: high tail latency or elevated p9Xs.

### Move initialization work out of the first invocation

Some work done on the first invocation can be moved up and completed ahead of time.

Any work that can be saved to disk, like
[downloading model weights](https://modal.com/docs/guide/model-weights),
should be done as early as possible. The results can be included in the
[container's Image](https://modal.com/docs/guide/images)
or saved to a
[Modal Volume](https://modal.com/docs/guide/volumes).

Some work is tricky to serialize, like spinning up a network connection or an inference server.
If you can move this initialization logic out of the function body and into the global scope or a
[container `enter` method](https://modal.com/docs/guide/lifecycle-functions#enter),
you can move this work into the warm up period.
Containers will not be considered warm until all `enter` methods have completed,
so no inputs will be routed to containers that have yet to complete this initialization.

For more on how to use `enter` with machine learning model weights, see
[this guide](https://modal.com/docs/guide/model-weights).

Note that `enter` doesn't get rid of the latency --
it just moves the latency to the warm up period,
where it can be handled by
[running more warm containers](#run-more-warm-containers).

### Share initialization work across cold starts with memory snapshots

Cold starts can also be made faster by using memory snapshots.

Invocations of a Function after the first
are faster in part because the memory is already populated
with values that otherwise need to be computed or read from disk,
like the contents of imported libraries.

Memory snapshotting captures the state of a container's memory
at user-controlled points after it has been warmed up
and reuses that state in future boots, which can substantially
reduce cold start latency penalties and warm up period duration.

Refer to the [memory snapshot](https://modal.com/docs/guide/memory-snapshot)
guide for details.

### Optimize initialization code

Sometimes, there is nothing to be done but to speed this work up.

Here, we share specific patterns that show up in optimizing initialization
in Modal Functions.

#### Load multiple large files concurrently

Often Modal applications need to read large files into memory (eg. model
weights) before they can process inputs. Where feasible these large file
reads should happen concurrently and not sequentially. Concurrent IO takes
full advantage of our platform's high disk and network bandwidth
to reduce latency.

One common example of slow sequential IO is loading multiple independent
Huggingface `transformers` models in series.

```python notest
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
model_a = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor_a = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model_b = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
processor_b = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
```

The above snippet does four `.from_pretrained` loads sequentially.
None of the components depend on another being already loaded in memory, so they
can be loaded concurrently instead.

They could instead be loaded concurrently using a function like this:

```python notest
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration

def load_models_concurrently(load_functions_map: dict) -> dict:
    model_id_to_model = {}
    with ThreadPoolExecutor(max_workers=len(load_functions_map)) as executor:
        future_to_model_id = {
            executor.submit(load_fn): model_id
            for model_id, load_fn in load_functions_map.items()
        }
        for future in as_completed(future_to_model_id.keys()):
            model_id_to_model[future_to_model_id[future]] = future.result()
    return model_id_to_model

components = load_models_concurrently({
    "clip_model": lambda: CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
    "clip_processor": lambda: CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
    "blip_model": lambda: BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large"),
    "blip_processor": lambda: BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
})
```

If performing concurrent IO on large file reads does _not_ speed up your cold
starts, it's possible that some part of your function's code is holding the
Python [GIL](https://wiki.python.org/moin/GlobalInterpreterLock) and reducing
the efficacy of the multi-threaded executor.

#### Memory Snapshot

# Memory Snapshot

Modal can save the state of your Function's memory right after initialization and restore it directly later, skipping initialization work.

These "memory snapshots" can dramatically improve cold start performance for Modal Functions.

During initialization, your code might read many files from the file system, which is quite expensive.
For example, the `torch` package is [hundreds of MiB](https://pypi.org/project/torch/#files) and requires over 20,000 file operations to load!
Such Functions typically start several times faster with memory snapshots enabled.

The memory snapshot feature has two variants. GPU memory snapshots (alpha) provide full GPU access before the snapshot is taken, while CPU memory snapshots do not.

## CPU Memory Snapshot

CPU memory snapshots capture the state of a container and save it to disk. This saved snapshot can then be used to quickly restore new containers to the exact same state.

### Basic usage

You can enable memory snapshots for your Function with the `enable_memory_snapshot=True` parameter:

```python
@app.function(enable_memory_snapshot=True)
def my_func():
    print("hello")
```

Then deploy the App with `modal deploy`. Memory snapshots are created only for deployed Apps.

When using classes decorated with [`@cls`](https://modal.com/docs/guide/lifecycle-functions), [`@modal.enter()`](https://modal.com/docs/reference/modal.enter) hooks are not included in the snapshot by default. Add `snap=True` to include them:

```python
@app.cls(enable_memory_snapshot=True)
class MyCls:
    @modal.enter(snap=True)
    def load(self):
        ...
```

Any code executed in global scope, such as top-level imports, will also be captured by the memory snapshot.

### CPU memory snapshots for GPU workloads

CPU memory snapshots don't support direct GPU memory capture, but GPU Functions can still benefit
from memory snapshots through a two-stage initialization process. This involves refactoring
your initialization code to run across two separate `@modal.enter` functions: one that runs before
creating the snapshot (`snap=True`), and one that runs after restoring from the
snapshot (`snap=False`). Load model weights onto CPU memory in the `snap=True`
method, and then move the weights onto GPU memory in the `snap=False` method.
Here's an example using the `sentence-transformers` package:

```python
import modal

image = modal.Image.debian_slim().pip_install("sentence-transformers")
app = modal.App("sentence-transformers", image=image)

with image.imports():
    from sentence_transformers import SentenceTransformer

model_vol = modal.Volume.from_name("sentence-transformers-models", create_if_missing=True)

@app.cls(gpu="a10g", volumes={"/models": model_vol}, enable_memory_snapshot=True)
class Embedder:
    model_id = "BAAI/bge-small-en-v1.5"

    @modal.enter(snap=True)
    def load(self):
        # Create a memory snapshot with the model loaded in CPU memory.
        self.model = SentenceTransformer(f"/models/{self.model_id}", device="cpu")

    @modal.enter(snap=False)
    def setup(self):
        self.model.to("cuda")  # Move the model to a GPU!

    @modal.method()
    def run(self, sentences:list[str]):
        embeddings = self.model.encode(sentences, normalize_embeddings=True)
        print(embeddings)

@app.local_entrypoint()
def main():
    Embedder().run.remote(sentences=["what is the meaning of life?"])

if __name__ == "__main__":
    cls = modal.Cls.from_name("sentence-transformers", "Embedder")
    cls().run.remote(sentences=["what is the meaning of life?"])
```

Even without GPU snapshotting, this workaround reduces the time it takes for `Embedder.run`
to startup by about 3x, from ~6 seconds down to just ~2 seconds.

### GPU availability during the memory snapshot phase

If you are using the GPU memory snapshot feature (`enable_gpu_snapshot`), then
GPUs are available within `@enter(snap=True)`.

If you are using memory snapshots _without_ `enable_gpu_snapshot`, then it's important
to note that GPUs will not be available within the `@enter(snap=True)` method.

```python
import modal
app = modal.App(image=modal.Image.debian_slim().pip_install("torch"))
@app.cls(enable_memory_snapshot=True, gpu="A10")
class GPUAvailability:
    @modal.enter(snap=True)
    def no_gpus_available_during_snapshots(self):
        import torch
        print(f"GPUs available: {torch.cuda.is_available()}")  # False

    @modal.enter(snap=False)
    def gpus_available_following_restore(self):
        import torch
        print(f"GPUs available: {torch.cuda.is_available()}")  # True

    @modal.method()
    def demo(self):
        print(f"GPUs available: {torch.cuda.is_available()}") # True
```

### Known limitations

The `torch.cuda` module has multiple functions which, if called during
snapshotting, will initialize CUDA as having zero GPU devices. Such functions
include `torch.cuda.is_available` and `torch.cuda.get_device_capability`.
If you're using a framework that calls these methods during its import phase,
it may not be compatible with memory snapshots. The problem can manifest as
confusing "cuda not available" or "no CUDA-capable device is detected" errors.

We have found that importing PyTorch twice solves the problem in some cases:

```python

@app.cls(enable_memory_snapshot=True, gpu="A10")
class GPUAvailability:
    @modal.enter(snap=True)
    def pre_snap(self):
        import torch
        ...
    @modal.enter(snap=False)
    def post_snap(self):
        import torch   # re-import to re-init GPU availability state
        ...
```

In particular, `xformers` is known to call `torch.cuda.get_device_capability` on
import, so if it is imported during snapshotting it can unhelpfully initialize
CUDA with zero GPUs. The
[workaround](https://github.com/facebookresearch/xformers/issues/1030) for this
is to set the `XFORMERS_ENABLE_TRITON` environment variable to `1` in your `modal.Image`.

```python
image = modal.Image.debian_slim().pip_install("xformers>=0.28")  # for instance
image = image.env({"XFORMERS_ENABLE_TRITON": "1"})
```

## GPU Memory Snapshot

With our experimental GPU memory snapshot feature, we are able to capture the entire GPU state too.
This makes for simpler initialization logic and even faster cold starts.

Pass the additional option `experimental_options={"enable_gpu_snapshot": True}` to your Function or class
to enable GPU snapshotting. These functions have full GPU and CUDA access.

```python
@app.function(
    gpu="a10",
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
def my_gpu_func():
    import torch
    print(f"GPUs available: {torch.cuda.is_available()}")  # True
```

Here's what the above `SentenceTransformer` example looks like with GPU memory snapshot enabled:

```python notest
@app.cls(
    gpu="a10g",
    volumes={"/models": model_vol},
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True}
)
class Embedder:
    model_id = "BAAI/bge-small-en-v1.5"

    @modal.enter(snap=True)
    def load(self):
        # Create a memory snapshot with the model loaded in GPU memory.
        self.model = SentenceTransformer(f"/models/{self.model_id}", device="cuda")
```

To achieve even faster cold starts, we recommend warming up your model by running a few forward passes on sample data
in the `@enter(snap=True)` method.

Refer to the code sample [here](https://modal.com/docs/examples/gpu_snapshot) for a more complete example. Our
[blog post](https://modal.com/blog/gpu-mem-snapshots) also provides more useful details.

### Known limitations

GPU memory snapshots are in _alpha_.
[We've seen](https://modal.com/blog/gpu-mem-snapshots) that they can massively reduce cold boot time
but we are still exploring their limitations. Try it for yourself and let us know how it goes!

## Memory Snapshot FAQ

### When are snapshots updated?

Redeploying your Function with new configuration (e.g. a [new GPU type](https://modal.com/docs/guide/gpu))
or new code will cause previous snapshots to become obsolete.
Subsequent invocations to the new Function version will automatically create new snapshots with the new configuration and code.

Changes to [Modal Volumes](https://modal.com/docs/guide/volumes) do not cause snapshots to update.
Deleting files in a Volume used during restore will cause restore failures.

### I haven't changed my Function. Why do I still see snapshots being created sometimes?

Modal recaptures snapshots to keep up with the platform's latest runtime and security changes.

Additionally, you may observe your Function being memory
snapshot multiple times during its first few invocations. This happens because
memory snapshots are specific to the underlying worker type that created them (e.g. low-level processor details),
and Modal Functions run across a handful of worker types.

Snapshots may add a small amount of latency to Function initialization.

CPU-only Functions need around 6 snapshots for full coverage, and Functions targeting a specific
GPU (e.g. A100) need 2-3.

### How do snapshots handle randomness?

If your application depends on uniqueness of state, you must evaluate your
Function code and verify that it is resilient to snapshotting operations. For
example, if a variable is randomly initialized and snapshotted, that variable
will be identical after every restore, possibly breaking uniqueness expectations
of the proceeding Function code.

#### Geographic latency

# Geographic Latency

Modal's worker cluster is multi-cloud and multi-region. The vast majority of workers are located
in the continental USA, but we do run workers in Europe and Asia.

Modal's control plane is hosted in Virginia, USA (`us-east-1`).

Any time data needs to travel between the Modal client, our control plane servers, and our workers
latency will be incurred. [Cloudping.co](https://www.cloudping.co) provides good estimates on the
significance of the latency between regions. For example, the roundtrip latency between AWS `us-east-1` (Virginia, USA) and
`us-west-1` (California, USA) is around 60ms.

You can observe the location identifier of a container [via an environment variable](https://modal.com/docs/guide/environment_variables).
Logging this environment variable alongside latency information can reveal when geography is impacting your application
performance.

## Region selection

In cases where low-latency communication is required between your container and a network dependency (e.g a database),
it is useful to ensure that Modal schedules your container in only regions geographically proximate to that dependency.
For example, if you have an AWS RDS database in Virginia, USA (`us-east-1`), ensuring your Modal containers are also scheduled in Virginia
means that network latency between the container and the database will be less than 5 milliseconds.

For more information, please see [Region selection](https://modal.com/docs/guide/region-selection).

### Reliability and robustness

#### Failures and retries

# Failures and retries

When you call a function over a sequence of inputs with
[Function.map()](https://modal.com/docs/guide/scale#parallel-execution-of-inputs), sometimes
errors can happen during function execution. Exceptions from within the remote
function are propagated to the caller, so you can handle them with a
`try-except` statement (refer to
[section on custom types](https://modal.com/docs/guide/troubleshooting#custom-types-defined-in-__main__)
for more on how to catch user-defined exceptions):

```python
@app.function()
def f(i):
    raise ValueError()

@app.local_entrypoint()
def main():
    try:
        for _ in f.map([1, 2, 3]):
            pass
    except ValueError:
        print("Exception handled")
```

## Function retries

You can configure Modal to automatically retry function failures if you set the
`retries` option when declaring your function:

```python
@app.function(retries=3)
def my_flaky_function():
    pass
```

When used with `Function.map()`, each input is retried up to the max number of
retries specified.

The basic configuration shown provides a fixed 1s delay between retry attempts.
For fine-grained control over retry delays, including exponential backoff
configuration, use [`modal.Retries`](https://modal.com/docs/reference/modal.Retries).

To treat exceptions as successful results and aggregate them in the results list instead, pass in [`return_exceptions=True`](https://modal.com/docs/guide/scale#exceptions).

## Container crashes

If a `modal.Function` container crashes (either on start-up, e.g. while handling imports in global scope, or during execution, e.g. an out-of-memory error), Modal will reschedule the container and any work it was currently assigned.

For [ephemeral apps](https://modal.com/docs/guide/apps#ephemeral-apps), container crashes will be retried until a failure rate is exceeded, after which all pending inputs will be failed and the exception will be propagated to the caller.

For [deployed apps](https://modal.com/docs/guide/apps#deployed-apps), container crashes will be retried indefinitely, so as to not disrupt service. Modal will instead apply a crash-loop backoff and the rate of new container creation for the function will be slowed down. Crash-looping containers are displayed in the app dashboard.

#### Preemption

# Preemption

All Modal Functions are subject to preemption. If a preemption event interrupts
a running Function, Modal will gracefully terminate the Function and restart it
on the same input.

Preemptions are rare, but it is always possible that your Function is
interrupted. Long-running Functions such as model training Functions should take
particular care to tolerate interruptions, as likelihood of interruption increases
with Function run duration.

## Preparing for interruptions

Design your applications to be fault and preemption tolerant. Modal will send an
interrupt signal to your container when preemption occurs. This will cause the
Function's [exit handler](https://modal.com/docs/guide/lifecycle-functions#exit) to run, which
can perform any cleanup within its grace period.

Other best practices for handling preemptions include:

- Divide long-running operations into small tasks or use checkpoints so that you
  can save your work frequently.
- Ensure preemptible operations are safely retryable (ie. idempotent).

## Running uninterruptible Functions

We currently don't have a way for Functions to avoid the possibility of
interruption, but it's a planned feature. If you require Functions guaranteed to
run without interruption, please reach out!

## Uninterruptible Sandboxes

Modal Sandboxes are not subject to preemption, except in the case where a `region`, `cloud`, or `gpu`
requirement is specified. This is because of availability and scheduling latency constraints.

#### Timeouts

# Timeouts

All Modal [Function](https://modal.com/docs/reference/modal.Function) executions have a default
execution timeout of 300 seconds (5 minutes), but users may specify timeout
durations between 1 second and 24 hours.

```python
import time

@app.function()
def f():
    time.sleep(599)  # Timeout!

@app.function(timeout=600)
def g():
    time.sleep(599)
    print("*Just* made it!")
```

The timeout duration is a measure of a Function's _execution_ time. It does not
include scheduling time or any other period besides the time your code is
executing in Modal. This duration is also per execution attempt, meaning
Functions configured with [`modal.Retries`](https://modal.com/docs/reference/modal.Retries) will
start new execution timeouts on each retry. For example, an infinite-looping
Function with a 100 second timeout and 3 allowed retries will run for least 400
seconds within Modal.

### Container startup timeout

A Function's `startup_timeout` configures the container's _startup_ time. Your container
may be taking a long time to startup because it is loading large data, initializing a
large model or importing many packages. In these cases, you can extend the
`startup_timeout` of your Function.

```python
@app.cls(startup_timeout=30, timeout=10)
class MyFunction:
    @modal.enter()
    def startup(self):
        time.sleep(20)

    @modal.method()
    def f(self):
        time.sleep(1)
```

`startup_timeout` was added in v1.1.4. Prior to v1.1.4, `timeout` configures the
_execution_ time and _startup_ time. If `startup_timeout` is not set, `timeout` will
still configure both times.

## Handling timeouts

After exhausting any specified retries, a timeout in a Function will produce a
`modal.exception.FunctionTimeoutError` which you may catch in your code.

```python
import modal.exception

@app.function(timeout=100)
def f():
    time.sleep(200)  # Timeout!

@app.local_entrypoint()
def main():
    try:
        f.remote()
    except modal.exception.FunctionTimeoutError:
        ... # Handle the timeout.
```

## Timeout accuracy

Functions will run for _at least_ as long as their timeout allows, but they may
run a handful of seconds longer. If you require accurate and precise timeout
durations on your Function executions, it is recommended that you implement
timeout logic in your user code.

#### GPU health

# GPU Health

Modal constantly monitors host GPU health, draining Workers with critical issues
and surfacing warnings for customer triage.

Application level observability of GPU health is facilitated by [metrics](https://modal.com/docs/guide/gpu-metrics) and event logging to container log streams.

## `[gpu-health]` logging

Containers with attached NVIDIA GPUs are connected to our `gpu-health` monitoring system
and receive event logs which originate from either application software behavior, system software behavior, or hardware failure.

These logs are in the following format: `[gpu-health] [LEVEL] GPU-[UUID]: EVENT_TYPE: MSG`

- `gpu-health`: Name indicating the source is Modal's observability system.
- `LEVEL`: Represents the severity level of the log message.
- `GPU_UUID`: A unique identifier for the GPU device associated with the event, if any.
- `EVENT_TYPE`: The type of event source. Modal monitors for multiple types of errors,
  including Xid, SXid, and uncorrectable ECC. See below for more details.
- `MSG`: The message component is either the original message taken from the event source, or a description provided by Modal of the problem.

## Level

The severity level may be `CRITICAL` or `WARN`. Modal automatically responds to `CRITICAL` level events by draining the underlying Worker and migrating customer containers.
`WARN` level logs may be benign or indication of an application or library bug. No automatic action is taken by our system for warnings.

## Handling Application-level health issues

As noted above, Modal will automatically respond to critical GPU events, but warning level events can still
be associated with application exceptions. Applications should catch exceptions caused by GPU-related faults
and call `modal.experimental.stop_fetching_inputs()`:

<!-- TODO: Migrate snippet to modal.Container when it's shipped. ref: https://modal-com.slack.com/archives/C056CGAANRM/p1756931590088119 -->

```python
import modal.experimental
...

@app.function(gpu="H100")
def demo():
    try:
        ... # code which may hit GPU fault (e.g. illegal memory access)
    except RuntimeError:
        modal.experimental.stop_fetching_inputs()
        return
```

## Xid & SXid

The Xid message is an error report from the NVIDIA driver. The SXid, or "Switch Xid" is a report for the NVSwitch component used in GPU-to-GPU communication, and is thus only relevant in multi-GPU containers.

A classic critical Xid error is the 'fell of the bus' report, code 79. The `gpu-health` event log looks like this:

```
[gpu-health] [CRITICAL] GPU-1234: XID: NVRM: Xid (PCI:0000:c6:00): 79, pid=1101234, name=nvc:[driver], GPU has fallen off the bus.
```

There are over 100 Xid codes and they are of highly varying frequency, severity, and specificity.
[NVIDIA's official documentation](https://docs.nvidia.com/deploy/xid-errors/index.html) provides limited information, so
we maintain our own tabular information below.

#### Troubleshooting

# Troubleshooting

## "Command not found" errors

If you installed Modal but you're seeing an error like
`modal: command not found` when trying to run the CLI, this means that the
installation location of Python package executables ("binaries") are not present
on your system path. This is a common problem; you need to reconfigure your
system's environment variables to fix it.

One workaround is to use `python -m modal.cli` instead of `modal`. However, this
is just a patch. There's no single solution for the problem because Python
installs dependencies on different locations depending on your environment. See
this [popular StackOverflow question](https://stackoverflow.com/q/35898734) for
pointers on how to resolve your system path issue.

## Custom types defined in `__main__`

Modal currently uses [cloudpickle](https://github.com/cloudpipe/cloudpickle) to
transfer objects returned or exceptions raised by functions that are executed in
Modal. This gives a lot of flexibility and support for custom data types.

However, any types that are declared in your Python entrypoint file (The one you
call on the command line) will currently be _redeclared_ if they are returned
from Modal functions, and will therefore have the same structure and type name
but not maintain class object identity with your local types. This means that
you _can't_ catch specific custom exception classes:

```python
import modal
app = modal.App()

class MyException(Exception):
    pass

@app.function()
def raise_custom():
    raise MyException()

@app.local_entrypoint()
def main():
    try:
        raise_custom.remote()
    except MyException:  # this will not catch the remote exception
        pass
    except Exception:  # this will catch it instead, as it's still a subclass of Exception
        pass
```

Nor can you do object equality checks on `dataclasses`, or `isinstance` checks:

```python
import modal
import dataclasses

@dataclasses.dataclass
class MyType:
    foo: int

app = modal.App()

@app.function()
def return_custom():
    return MyType(foo=10)

@app.local_entrypoint()
def main():
    data = return_custom.remote()
    assert data == MyType(foo=10)  # false!
    assert data.foo == 10  # true!, the type still has the same fields etc.
    assert isinstance(data, MyType)  # false!
```

If this is a problem for you, you can easily solve it by moving your custom type
definitions to a separate Python file from the one you trigger to run your Modal
code, and import that file instead.

```python
# File: my_types.py
import dataclasses

@dataclasses.dataclass
class MyType:
    foo: int
```

```python notest
# File: modal_script.py
import modal
from my_types import MyType

app = modal.App()

@app.function()
def return_custom():
    return MyType(foo=10)

@app.local_entrypoint()
def main():
    data = return_custom.remote()
    assert data == MyType(foo=10)  # true!
    assert isinstance(data, MyType)  # true!
```

## Function side effects

The same container _can_ be reused for multiple invocations of the same function
within an app. This means that if your function has side effects like modifying
files on disk, they may or may not be present for subsequent calls to that
function. You should not rely on the side effects to be present, but you might
have to be careful so they don't cause problems.

For example, if you create a disk-backed database using sqlite3:

```python
import modal
import sqlite3

app = modal.App()

@app.function()
def db_op():
    db = sqlite3("db_file.sqlite3")
    db.execute("CREATE TABLE example (col_1 TEXT)")
    ...
```

This function _can_ (but will not necessarily) fail on the second invocation
with an

`OperationalError: table foo already exists`

To get around this, take care to either clean up your side effects (e.g.
deleting the db file at the end your function call above) or make your functions
take them into consideration (e.g. adding an
`if os.path.exists("db_file.sqlite")` condition or randomize the filename
above).

## Heartbeat timeout

The Modal client in `modal.Function` containers runs a heartbeat loop that the host uses to healthcheck the container's main process.
If the container stops heartbeating for a long period (minutes) the container will be terminated due to a `heartbeat timeout`, which is displayed in logs.

Container heartbeat timeouts are rare, and typically caused by one of two application-level sources:

- [Global Interpreter Lock](https://wiki.python.org/moin/GlobalInterpreterLock) is held for a long time, stopping the heartbeat thread from making progress. [py-spy](https://github.com/benfred/py-spy?tab=readme-ov-file#how-does-gil-detection-work) can detect GIL holding. We include `py-spy` [automatically in `modal shell`](https://modal.com/docs/guide/developing-debugging#debug-shells) for convenience. A quick fix for GIL holding is to run the code which holds the GIL [in a subprocess](https://docs.python.org/3/library/multiprocessing.html#the-process-class).
- Container process initiates shutdown, intentionally stopping the heartbeats, but it does not complete shutdown.

In both cases [turning on debug logging](https://modal.com/docs/guide/developing-debugging#debug-logs) will help diagnose the issue.

## `413 Content Too Large` errors

If you receive a `413 Content Too Large` error, this might be because you are
hitting our gRPC payload size limits.

The size limit is currently 100MB.

## `403` errors when connecting to GCP services.

GCP will sometimes return 403 errors to Modal when connecting directly to GCP
cloud services like Google Cloud Storage. This is a known issue.

The workaround is to pin the `cloud` parameter in the
[`@app.function`](https://modal.com/docs/reference/modal.App#function) or
[`@app.cls`](https://modal.com/docs/reference/modal.App#cls).

For example:

```python
@app.function(cloud="gcp")
def f():
    ...
```

```python
@app.cls(cloud="gcp")
class MyClass:
    ...
```

## Outdated kernel version (4.4.0)

Our secure runtime [reports a misleadingly old](https://github.com/google/gvisor/issues/11117) kernel version, 4.4.0.
Certain software libraries will detect this and report a warning. These warnings can be ignored because the runtime
actually implements Linux kernel features from versions 5.15+.

If the outdated kernel version reporting creates errors in your application please contact us [in our Slack](https://modal.com/slack).

### Security and privacy

# Security and privacy at Modal

The document outlines Modal's security and privacy commitments.

## Application security (AppSec)

AppSec is the practice of building software that is secure by design, secured
during development, secured with testing and review, and deployed securely.

- We build our software using memory-safe programming languages, including Rust
  (for our worker runtime and storage infrastructure) and Python (for our API
  servers and Modal client).
- Software dependencies are audited by Github's Dependabot.
- We make decisions that minimize our attack surface. Most interactions with
  Modal are well-described in a gRPC API, and occur through
  [`modal`](https://pypi.org/project/modal), our open-source command-line tool
  and Python client library.
- We have automated synthetic monitoring test applications that continuously
  check for network and application isolation within our runtime.
- We use HTTPS for secure connections. Modal forces HTTPS for all services using
  TLS (SSL), including our public website and the Dashboard to ensure secure
  connections. Modal's [client library](https://pypi.org/project/modal) connects
  to Modal's servers over TLS and verify TLS certificates on each connection.
- All user data is encrypted in transit and at rest.
- All public Modal APIs use
  [TLS 1.3](https://datatracker.ietf.org/doc/html/rfc8446), the latest and
  safest version of the TLS protocol.
- Internal code reviews are performed using a modern, PR-based development
  workflow (Github), and engage external penetration testing firms to assess our
  software security.

## Corporate security (CorpSec)

CorpSec is the practice of making sure Modal employees have secure access to
Modal company infrastructure, and also that exposed channels to Modal are
secured. CorpSec controls are the primary concern of standards such as SOC2.

- Access to our services and applications is gated on a SSO Identity Provider
  (IdP).
- We mandate phishing-resistant multi-factor authentication (MFA) in all
  enrolled IdP accounts.
- We regularly audit access to internal systems.
- Employee laptops are protected by full disk encryption using FileVault2, and
  managed by Secureframe MDM.

## Network and infrastructure security (InfraSec)

InfraSec is the practice of ensuring a hardened, minimal attack surface for
components we deploy on our network.

- Modal uses logging and metrics observability providers, including Datadog and
  Sentry.io.
- Compute jobs at Modal are containerized and virtualized using
  [gVisor](https://github.com/google/gvisor), the sandboxing technology
  developed at Google and used in their _Google Cloud Run_ and _Google
  Kubernetes Engine_ cloud services.
- We conduct annual business continuity and security incident exercises.
<!-- TODO: we don't yet encrypt network file system data. "Customer information on databases and volumes at Modal is encrypted with the Linux LUKS block storage encryption secrets." -->

## Vulnerability remediation

Security vulnerabilities directly affecting Modal's systems and services will be
patched or otherwise remediated within a timeframe appropriate for the severity
of the vulnerability, subject to the public availability of a patch or other
remediation mechanisms.

If there is a CVSS severity rating accompanying a vulnerability disclosure, we
rely on that as a starting point, but may upgrade or downgrade the severity
using our best judgement.

### Severity timeframes

- **Critical:** 24 hours
- **High:** 1 week
- **Medium:** 1 month
- **Low:** 3 months
- **Informational:** 3 months or longer

## Shared responsibility model

Modal prioritizes the integrity, security, and availability of customer data. Under our shared responsibility model, customers also have certain responsibilities regarding data backup, recovery, and availability.

1. **Data backup**: Customers are responsible for maintaining backups of their data. Performing daily backups is recommended. Customers must routinely verify the integrity of their backups.
2. **Data recovery**: Customers should maintain a comprehensive data recovery plan that includes detailed procedures for data restoration in the event of data loss, corruption, or system failure. Customers must routinely test their recovery process.
3. **Availability**: While Modal is committed to high service availability, customers must implement contingency measures to maintain business continuity during service interruptions. Customers are also responsible for the reliability of their own IT infrastructure.
4. **Security measures**: Customers must implement appropriate security measures, such as encryption and access controls, to protect their data throughout the backup, storage, and recovery processes. These processes must comply with all relevant laws and regulations.

## SOC 2

We have successfully completed a [System and Organization Controls (SOC) 2 Type 2
audit](https://modal.com/blog/soc2type2). Go to our [Security Portal](https://trust.modal.com) to request access to the report.

## HIPAA

HIPAA, which stands for the Health Insurance Portability and Accountability Act, establishes a set of standards that protect health information, including individuals’ medical records and other individually identifiable health information. HIPAA guidelines apply to both covered entities and business associates—of which Modal is the latter if you are processing PHI on Modal.

Modal's services can be used in a HIPAA compliant manner. It is important to note that unlike other security standards, there is no officially recognized certification process for HIPAA compliance. Instead, we demonstrate our compliance with regulations such as HIPAA via the practices outlined in this doc, our technical and operational security measures, and through official audits for standards compliance such as SOC 2 certification.

To use Modal services for HIPAA-compliant workloads, a Business Associate Agreement (BAA) should be established with us prior to submission of any PHI. This is available on our Enterprise plan. Contact us at security@modal.com to get started. At the moment, [Volumes v1](https://modal.com/docs/guide/volumes), [Images](https://modal.com/docs/guide/images) (persistent storage), [memory snapshots](https://modal.com/docs/guide/memory-snapshot), and user code are out of scope of the commitments within our BAA, so PHI should not be used in those areas of the product.

[Volumes v2](https://modal.com/docs/guide/volumes#volumes-v2) are HIPAA compliant.

## PCI

_Payment Card Industry Data Security Standard_ (PCI) is a standard that defines
the security and privacy requirements for payment card processing.

Modal uses [Stripe](https://stripe.com) to securely process transactions and
trusts their commitment to best-in-class security. We do not store personal
credit card information for any of our customers. Stripe is certified as "PCI
Service Provider Level 1", which is the highest level of certification in the
payments industry.

## Bug bounty program

Keeping user data secure is a top priority at Modal. We welcome contributions
from the security community to identify vulnerabilities in our product and
disclose them to us in a responsible manner. We offer rewards ranging from $100
to $1000+ depending on the severity of the issue discovered. To participate,
please send a report of the vulnerability to security@modal.com.

## Data privacy

Modal will never access or use:

- your source code.
- the inputs (function arguments) or outputs (function return values) to your Modal Functions.
- any data you store in Modal, such as in Images or Volumes.

Inputs (function arguments) and outputs (function return values) are deleted from our system after a max TTL of 7 days.

App logs and metadata are stored on Modal. Modal will not access this data
unless permission is granted by the user to help with troubleshooting.

## Questions?

[Email us!](mailto:security@modal.com)

### Integrations

#### Using OIDC to authenticate with external services

# Using OIDC to authenticate with external services

Your Functions in Modal may need to access external resources like S3 buckets.
Traditionally, you would need to store long-lived credentials in Modal Secrets
and reference those Secrets in your function code. With the Modal OIDC
integration, you can instead use automatically-generated identity
tokens to authenticate to external services.

## How it works

[OIDC](https://auth0.com/docs/authenticate/protocols/openid-connect-protocol) is
a standard protocol for authenticating users between systems. In Modal, we use
OIDC to generate short-lived tokens that external services can use to verify
that your function is authenticated.

The OIDC integration has two components: the discovery document and the generated
tokens.

The [OIDC discovery document](https://swagger.io/docs/specification/v3_0/authentication/openid-connect-discovery/)
describes how our OIDC server is configured. It primarily includes the supported
[claims](https://developer.okta.com/blog/2017/07/25/oidc-primer-part-1) and the [keys](https://auth0.com/docs/secure/tokens/json-web-tokens/json-web-key-sets)
we use to sign tokens. Discovery documents are always hosted at `/.well-known/openid-configuration`, and
you can view ours at <https://oidc.modal.com/.well-known/openid-configuration>.

The generated tokens are [JWTs](https://jwt.io/) signed by Modal using the keys described in the
discovery document. These tokens contain the full identity of the Function
in the `sub` claim, and they use custom claims to make this information more
easily accessible. See our [discovery document](https://oidc.modal.com/.well-known/openid-configuration)
for a full list of claims.

Generated tokens are injected into your Function's containers via the `MODAL_IDENTITY_TOKEN`
environment variable. Below is an example of what claims might be included in a token:

```json
{
  "sub": "modal:workspace_id:ac-12345abcd:environment_name:modal-examples:app_name:oidc-token-test:function_name:jwt_return_func:container_id:ta-12345abcd",
  "aud": "oidc.modal.com",
  "exp": 1732137751,
  "iat": 1731964951,
  "iss": "https://oidc.modal.com",
  "jti": "31f92dca-e847-4bc9-8d15-9f234567a123",
  "workspace_id": "ac-12345abcd",
  "environment_id": "en-12345abcd",
  "environment_name": "modal-examples",
  "app_id": "ap-12345abcd",
  "app_name": "oidc-token-test",
  "function_id": "fu-12345abcd",
  "function_name": "jwt_return_func",
  "container_id": "ta-12345abcd"
}
```

### Key thumbprints

RSA keys have [thumbprints](https://connect2id.com/products/nimbus-jose-jwt/examples/jwk-thumbprints). You
can use these thumbprints to verify that the keys in our discovery document are
genuine. This protects against potential Man in the Middle (MitM) attacks, although
our required use of HTTPS mitigates this risk.

If you'd like to have the extra security of verifying the thumbprints, you can
use the following command to print the thumbprints for the keys in our
discovery document:

```bash
$ openssl s_client -connect oidc.modal.com:443 < /dev/null 2>/dev/null | openssl x509 -fingerprint -noout | awk -F= '{print $2}' | tr -d ':'
F062F2151EDE30D1620B48B7AC91D66047D769D3
```

Note that these thumbprints may change over time as we rotate keys. We recommend
periodically checking for and updating your scripts with the new thumbprints.

### App name format

By default, Modal Apps can be created with arbitrary names. However, when using
OIDC, the App name has a stricter character set. Specifically, it must be 64
characters or less and can only include alphanumeric characters, dashes, periods,
and underscores. If these constraints are violated, the OIDC token will not be
injected into the container.

Note that these are the same constraints that are applied to [Deployed Apps](https://modal.com/docs/guide/managing-deployments).
This means that if an App is deployable, it will also be compatible with OIDC.

## Demo usage with AWS S3

To see how OIDC tokens can be used, we'll demo a simple Function that lists
objects in an S3 bucket.

### Step 0: Understand your OIDC claims

Before we can configure OIDC policies, we need to know what claims we can match
against. We can run a Function and inspect its claims to find out.

```python notest
app = modal.App("oidc-token-test")

jwt_image = modal.Image.debian_slim().pip_install("pyjwt")

@app.function(image=jwt_image)
def jwt_return_func():
    import jwt

    token = os.environ["MODAL_IDENTITY_TOKEN"]
    claims = jwt.decode(token, options={"verify_signature": False})
    print(json.dumps(claims, indent=2))

@app.local_entrypoint()
def main():
    jwt_return_func.remote()
```

Run the function locally to see its claims:

```bash
$ modal run oidc-token-test.py
{
  "sub": "modal:workspace_id:ac-12345abcd:environment_name:modal-examples:app_name:oidc-token-test:function_name:jwt_return_func:container_id:ta-12345abcd",
  "aud": "oidc.modal.com",
  "exp": 1732137751,
  "iat": 1731964951,
  "iss": "https://oidc.modal.com",
  "jti": "31f92dca-e847-4bc9-8d15-9f234567a123",
  "workspace_id": "ac-12345abcd",
  "environment_id": "en-12345abcd",
  "environment_name": "modal-examples",
  "app_id": "ap-12345abcd",
  "app_name": "oidc-token-test",
  "function_id": "fu-12345abcd",
  "function_name": "jwt_return_func",
  "container_id": "ta-12345abcd"
}
```

Now we can match off these claims to configure our OIDC policies.

### Step 1: Configure AWS to trust Modal's OIDC provider

We need to make AWS accept Modal identity tokens. To do this, we need to add
Modal's OIDC provider as a trusted entity in our AWS account.

```bash
aws iam create-open-id-connect-provider \
    --url https://oidc.modal.com \
    --client-id-list oidc.modal.com \
    # Optionally replace with the thumbprint from the discovery document.
    # Note that this may change over time as we rotate keys, and this argument
    # can be omitted if you'd prefer to rely on the HTTPS verification instead.
    --thumbprint-list "<thumbprint>"
```

This will trigger AWS to pull down our [JSON Web Key Set (JWKS)](https://auth0.com/docs/secure/tokens/json-web-tokens/json-web-key-sets)
and use it to verify the signatures of any tokens signed by Modal.

### Step 2: Create an IAM policy that can be used by Modal Functions

Let's create a simple IAM policy that allows listing objects in an S3 bucket.
Take the policy below and replace the bucket name with your own.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
      "Resource": ["arn:aws:s3:::fun-bucket", "arn:aws:s3:::fun-bucket/*"]
    }
  ]
}
```

### Step 3: Create an IAM role that can be assumed by Modal Functions

Now, we can create an IAM role that uses this policy. Visit the IAM console
to create this role. If you add this policy using the CLI, update the
OIDC provider ARN to match the one created in [Step 1](#step-1-configure-aws-to-trust-modals-oidc-provider).
Be sure to replace the Workspace ID placeholder with your own. You can find your Workspace ID
using the script from [Step 0](#step-0-understand-your-oidc-claims).

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::123456789abcd:oidc-provider/oidc.modal.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "oidc.modal.com:aud": "oidc.modal.com"
        },
        "StringLike": {
          "oidc.modal.com:sub": "modal:workspace_id:ac-12345abcd:*"
        }
      }
    }
  ]
}
```

Note how we use `workspace_id` to limit the scope of the role. This means that
the IAM role can only be assumed by Functions in your Workspace. You can further
limit this by specifying an Environment, App, or Function name.

Ideally, we would use the custom claims for role limiting. Unfortunately, AWS
does not support [matching on custom claims](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_iam-condition-keys.html#condition-keys-wif),
so we use the `sub` claim instead.

### Step 4: Use the OIDC token in your Function

The AWS SDKs have built-in support for OIDC tokens, so you can use them as
follows:

```python notest
import boto3

app = modal.App("oidc-token-test")

boto3_image = modal.Image.debian_slim().pip_install("boto3")

# Trade a Modal OIDC token for AWS credentials
def get_s3_client(role_arn):
    sts_client = boto3.client("sts")

    # Assume role with Web Identity
    credential_response = sts_client.assume_role_with_web_identity(
        RoleArn=role_arn, RoleSessionName="OIDCSession", WebIdentityToken=os.environ["MODAL_IDENTITY_TOKEN"]
    )

    # Extract credentials
    credentials = credential_response["Credentials"]
    return boto3.client(
        "s3",
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )

# List the contents of an S3 bucket
@app.function(image=boto3_image)
def list_bucket_contents(bucket_name, role_arn):
    s3_client = get_s3_client(role_arn)
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    for obj in response["Contents"]:
        print(f"- {obj['Key']} (Size: {obj['Size']} bytes)")

@app.local_entrypoint()
def main():
    # Replace with the role ARN and bucket name from step 2
    list_bucket_contents.remote("fun-bucket", "arn:aws:iam::123456789abcd:role/oidc_test_role")
```

Run the function locally to see the contents of the bucket:

```bash
$ modal run oidc-token-test.py
- test-file.txt (Size: 10 bytes)
```

## Demo usage with AWS Elastic Container Registry (ECR)

You can also use OIDC to authenticate [Private Registries](https://modal.com/docs/guide/existing-images) on AWS.

### Prerequisites

1. Configure AWS to trust Modal's OIDC provider ([Step 1 above](#step-1-configure-aws-to-trust-modals-oidc-provider))

2. [Create an AWS Policy with read-only ECR access](https://modal.com/docs/guide/existing-images#elastic-container-registry-ecr)

3. Create an IAM role that uses this policy ([Step 3 above](#step-3-create-an-iam-role-that-can-be-assumed-by-modal-functions))

### Test with a sample image

Create sample Dockerfile:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
CMD ["python3"]
```

Build and push the image to ECR:

```bash
# Login with the AWS CLI
aws ecr get-login-password --region [ECR_REGION] | docker login --username AWS --password-stdin [ECR_REPO_ARN]

# Build the Docker Image
docker build -t modal-oidc-test-image .

# Push the image to ECR
docker tag modal-oidc-test-image:latest [ECR_REPO_ARN]:latest
docker push [ECR_REPO_ARN]:latest
```

Test pulling the image from ECR:

```python
import modal

app = modal.App("image-from-ecr-test")
sample_image = modal.Image.from_aws_ecr(
    "[ECR_IMAGE_URI]", #eg. "12345678.dkr.ecr.us-east-1.amazonaws.com/repository:latest"
    secret=modal.Secret.from_dict(
        {
            "AWS_ROLE_ARN": "[IAM_ROLE_ARN]", # eg. "arn:aws:iam::123456789abcd:role/oidc_test_role"
            "AWS_REGION": "[ECR_REGION]", # eg. "us-east-1"
        }
    ),
)

@app.function(image=sample_image)
def hello():
    print("Hello, World!")
```

## Next steps

The OIDC integration can be used for much more than just AWS. With this same pattern,
you can configure automatic access to [Vault](https://developer.hashicorp.com/vault/docs/auth/jwt),
[GCP](https://cloud.google.com/identity-platform/docs/web/oidc), [Azure](https://learn.microsoft.com/en-us/entra/identity-platform/v2-protocols-oidc), and more.
At this time, OIDC-authenticated container image pulling is only support with AWS ECR.

#### Connecting Modal to your Datadog account

# Connecting Modal to your Datadog account

You can use the [Modal + Datadog Integration](https://docs.datadoghq.com/integrations/modal/)
to export Modal function logs to Datadog. You'll find the Modal Datadog
Integration available for install in the Datadog marketplace.

## What this integration does

This integration allows you to:

1. Export Modal audit logs in Datadog
2. Export Modal function logs to Datadog
3. Export container metrics to Datadog

## Installing the integration

1. Open the [Modal Tile](https://app.datadoghq.com/integrations?integrationId=modal) (or the EU tile [here](https://app.datadoghq.eu/integrations?integrationId=modal))
   in the Datadog integrations page
2. Click "Install Integration"
3. Click Connect Accounts to begin authorization of this integration.
   You will be redirected to log into Modal, and once logged in, you’ll
   be redirected to the Datadog authorization page.
4. Click "Authorize" to complete the integration setup

## Metrics

The Modal Datadog Integration will forward the following metrics to Datadog:

- `modal.cpu.utilization`
- `modal.memory.utilization`
- `modal.gpu.memory.utilization`
- `modal.gpu.compute.utilization`
- `modal.input_events.elapsed_time_us`
- `modal.input_events.successes`
- `modal.input_events.total_inputs`

`modal.input_events.successes` and `modal.input_events.total_inputs` can be used to measure the success rate of a certain function or app.

These metrics come free of charge and are tagged with `container_id`, `environment_name`,
`app_name`, `app_id`, `function_name`, `function_id`, `workspace_name`, and `workspace_id`.

## Structured logging

Logs from Modal are sent to Datadog in plaintext without any structured
parsing. This means that if you have custom log formats, you'll need to
set up a [log processing pipeline](https://docs.datadoghq.com/logs/log_configuration/pipelines/?tab=source)
in Datadog to parse them.

Modal passes log messages in the `.message` field of the log record. To
parse logs, you should operate over this field. Note that the Modal Integration
does set up some basic pipelines. In order for your pipelines to work, ensure
that your pipelines come before Modal's pipelines in your log settings.

## Cost Savings

The Modal Datadog Integration will forward all logs to Datadog which could be
costly for verbose apps. We recommend using either [Log Pipelines](https://docs.datadoghq.com/logs/log_configuration/pipelines/?tab=source)
or [Index Exclusion Filters](https://docs.datadoghq.com/logs/indexes/?tab=ui#exclusion-filters)
to filter logs before they are sent to Datadog.

The Modal Integration tags all logs with the `environment` attribute. The
simplest way to filter logs is to create a pipeline that filters on this
attribute and to isolate verbose apps in a separate environment.

## Uninstalling the integration

Once the integration is uninstalled, all logs will stop being sent to
Datadog, and authorization will be revoked.

1. Navigate to the [Modal metrics settings page](http://modal.com/settings/metrics)
   and select "Delete Datadog Integration".
2. On the Configure tab in the Modal integration tile in Datadog,
   click Uninstall Integration.
3. Confirm that you want to uninstall the integration.
4. Ensure that all API keys associated with this integration have been
   disabled by searching for the integration name on the [API Keys](https://app.datadoghq.com/organization-settings/api-keys?filter=Modal)
   page.

#### Connecting Modal to your OpenTelemetry provider

# Connecting Modal to your OpenTelemetry Provider

You can export Modal logs to your [OpenTelemetry](https://opentelemetry.io/docs/what-is-opentelemetry/)
provider using the Modal OpenTelemetry integration. This integration is compatible with
any observability provider that supports the OpenTelemetry HTTP APIs.

## What this integration does

This integration allows you to:

1. Export Modal audit logs to your provider
2. Export Modal function logs to your provider
3. Export container metrics to your provider

## Metrics

The Modal OpenTelemetry Integration will forward the following metrics to your provider:

- `modal.cpu.utilization`
- `modal.memory.utilization`
- `modal.gpu.memory.utilization`
- `modal.gpu.compute.utilization`
- `modal.input_events.elapsed_time_us`
- `modal.input_events.successes`
- `modal.input_events.total_inputs`

`modal.input_events.successes` and `modal.input_events.total_inputs` can be used to measure the success rate of a certain function or app.

These metrics are tagged with `container_id`, `environment_name`, `app_name`,
`app_id`, `function_name`, `function_id`, `workspace_name`, and `workspace_id`.

## Custom metrics (Beta)

The Modal OpenTelemetry Integration allows you to send custom metrics and spans to your provider.
To use this feature, export our collector environment variables. These configure the OpenTelemetry SDK
to send messages to our collector in HTTP format. You don't need to do this to get the
out-of-the-box metrics above, only for your own custom metrics.

```python
@app.function(
   secrets=[modal.Secret.from_dict({
      "OTEL_EXPORTER_OTLP_ENDPOINT": "otlp-collector.modal.local:4317",
      "OTEL_EXPORTER_OTLP_INSECURE": "true",
      "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
   })],
)
def custom_metrics():
   ...
```

All OpenTelemetry SDKs should pick this configuration up, and your custom metrics and spans will be
sent to your configured provider.

## Installing the integration

1. Find out the endpoint URL for your OpenTelemetry provider. This is the URL that
   the Modal integration will send logs to. Note that this should be the base URL
   of the OpenTelemetry provider, and not a specific endpoint. For example, for the
   [US New Relic instance](https://docs.newrelic.com/docs/opentelemetry/best-practices/opentelemetry-otlp/#configure-endpoint-port-protocol),
   the endpoint URL is `https://otlp.nr-data.net`, not `https://otlp.nr-data.net/v1/logs`.
2. Find out the API key or other authentication method required to send logs to your
   OpenTelemetry provider. This is the key that the Modal integration will use to authenticate
   with your provider. Modal can provide any key/value HTTP header pairs. For example, for
   [New Relic](https://docs.newrelic.com/docs/opentelemetry/best-practices/opentelemetry-otlp/#api-key),
   the header is `api-key`.
3. Create a new OpenTelemetry Secret in Modal with one key per header. These keys should be
   prefixed with `OTEL_HEADER_`, followed by the name of the header. The value of this
   key should be the value of the header. For example, for New Relic, an example Secret
   might look like `OTEL_HEADER_api-key: YOUR_API_KEY`. If you use the OpenTelemetry Secret
   template, this will be pre-filled for you.
4. Navigate to the [Modal metrics settings page](http://modal.com/settings/metrics) and configure
   the OpenTelemetry push URL from step 1 and the Secret from step 3.
5. Save your changes and use the test button to confirm that logs are being sent to your provider.
   If it's all working, you should see a `Hello from Modal! 🚀` log from the `modal.test_logs` service.

## Uninstalling the integration

Once the integration is uninstalled, all logs will stop being sent to
your provider.

1. Navigate to the [Modal metrics settings page](http://modal.com/settings/metrics)
   and disable the OpenTelemetry integration.

#### Okta SSO

# Okta SSO

## Prerequisites

- A Workspace that's on an [Enterprise](https://modal.com/pricing) plan
- Admin access to the Workspace you want to configure with Okta Single-Sign-On (SSO)
- Admin privileges for your Okta Organization

## Supported features

- IdP-initiated SSO
- SP-initiated SSO
- Just-In-Time account provisioning

For more information on the listed features, visit the
[Okta Glossary](https://help.okta.com/okta_help.htm?type=oie&id=ext_glossary).

## Configuration

### Read this before you enable "Require SSO"

Enabling "Require SSO" will force all users to sign in via Okta. Ensure that you
have admin access to your Modal Workspace through an Okta account before
enabling.

### Configuration steps

#### Step 1: Add Modal app to Okta Applications

1. Sign in to your Okta admin dashboard
2. Navigate to the Applications tab and click "Browse App Catalog".
   ![Okta browse application](../../assets/docs/okta-browse-applications.png)

3. Select "Modal" and click "Done".
4. Select the "Sign On" tab and click "Edit".
   ![Okta sign on edit](../../assets/docs/okta-sign-on-edit.png)
5. Fill out Workspace field to configure for your specific Modal workspace. See
   [Step 2](https://modal.com/docs/guide/okta-sso#step-2-link-your-workspace-to-okta-modal-application)
   if you're unsure what this is.
   ![Okta add workspace](../../assets/docs/okta-add-workspace-username.png)

#### Step 2: Link your Workspace to Okta Modal application

1. Navigate to your application on the Okta Admin page.
2. Copy the Metadata URL from the Okta Admin Console (It's under the "Sign On"
   tab). ![Okta metadata url](../../assets/docs/okta-metadata-url.png)

3. Sign in to https://modal.com and visit your [Workspace Management](https://modal.com/settings/workspace-management/identity-and-provisioning) page's `Identity and Provisioning` tab.
4. Paste the Metadata URL in the input and click "Save Changes"

#### Step 3: Assign users / groups and test the integration

1. Navigate back to your Okta application on the Okta Admin dashboard.
2. Click on the "Assignments" tab and add the appropriate people or groups.

![Okta Assign Users](../../assets/docs/okta-assign-people.png)

3. To test the integration, sign in as one of the users you assigned in the previous step.
4. Click on the Modal application on the Okta Dashboard to initiate Single Sign-On.

#### Notes

The following SAML attributes are used by the integration:

| Name      | Value          |
| --------- | -------------- |
| email     | user.email     |
| firstName | user.firstName |
| lastName  | user.lastName  |

## SP-initiated SSO

The sign-in process is initiated from https://modal.com/login/sso

1. Enter your workspace name in the input
2. Click "continue with SSO" to authenticate with Okta

#### Slack notifications (beta)

# Slack notifications (beta)

You can integrate your Modal Workspace with Slack to receive timely essential notifications.

## Prerequisites

- You are a [Workspace Manager](https://modal.com/docs/guide/workspaces#administrating-workspace-members) in the Modal Workspace you're installing the Slack integration in.
- You have permissions to install apps in your Slack workspace.

## Supported notifications

- Alerts for failed scheduled function runs.
- Alerts for crash-looping containers in a function.
- Alerts when any of your apps have client versions that are out of date.
- Alerts when you hit your GPU resource limits.

## Slack Permissions

The Modal Slack app requests the following permissions to integrate with Slack:

- Start direct messages with people
- Send messages as @modal
- Add shortcuts and/or slash commands that people can use
- View basic information about public channels in a workspace
- View basic information about private channels that Modal has been added to
- View basic information about direct messages that Modal has been added to
- View basic information about group direct messages that Modal has been added to
- View people in a workspace

## Configuration

### Step 1: Install the Slack integration

Visit the _Slack Integration_ section on your [settings](https://modal.com/settings/slack-integration) page in your Modal Workspace and click the **Add to Slack** button.

### Step 2: Invite the Modal app to your Slack channel

Navigate to the Slack channel and `/invite` the Modal app so that the app can post messages to the channel.

![Adding an app to Slack channel](https://modal-cdn.com/cdnbot/slack-invite-app_vpxfskj_f0dc9524.webp)

### Step 3: Add the Modal app to your Slack channel

Navigate to the Slack channel you want to add the Modal app to and click on the channel header. On the integrations tab you can add the Modal app.

![Add Modal app to Slack channel](../../assets/docs/slack-add-modal-app.jpg)

### Step 4: Use `/modal link` to link the Slack channel to your Modal Workspace

You'll be prompted to select the Workspace you want to link to the Slack channel. You can always unlink the Slack channel by visiting the _Slack Integration_ section on your [settings](https://modal.com/settings/slack-integration) page in your Modal Workspace.

### Workspace & account settings

#### Workspaces

# Workspaces

A **workspace** is an area where a user can deploy Modal apps and other
resources. There are two types of workspaces: personal and shared. After a new
user has signed up to Modal, a personal workspace is automatically created for
them. The name of the personal workspace is based on your GitHub username, but
it might be randomly generated if already taken or invalid.

To collaborate with others, a new shared workspace needs to be created.

## Create a Workspace

All additional workspaces are shared workspaces, meaning you can invite others
by email to collaborate with you. There are two ways to create a Modal workspace
on the [settings](https://modal.com/settings/workspaces) page.

![view of workspaces creation interface](https://modal-cdn.com/cdnbot/create-new-workspace-viewk0ka46_7_800f2053.webp)

1. Create from [GitHub organization](https://docs.github.com/en/organizations). Allows members in GitHub organization to auto-join the workspace.

2. Create from scratch. You can invite anyone to your workspace.

If you're interested in having a workspace associated with your Okta
organization, then check out our [Okta SSO docs](https://modal.com/docs/guide/okta-sso).

If you're interested in using SSO through Google or other providers, then please reach out to us at [support@modal.com](mailto:support@modal.com).

## Auto-joining a Workspace associated with a GitHub organization

Note: This is only relevant for Workspaces created from a GitHub organization.

Users can automatically join a Workspace on their [Workspace settings page](https://modal.com/settings/workspaces) if they are a member of the GitHub organization associated with the Workspace.

To turn off this functionality a Workspace Manager can disable it on the **Workspace Management** tab of their Workspace's settings page.

## Inviting new Workspace members

To invite a new Workspace member, you can visit the [settings](https://modal.com/settings) page
and navigate to the members tab for the appropriate workspace.

You can either send an email invite or share an invite link. Both existing Modal
users and non-existing users can use the links to join your workspace. If they
are a new user a Modal account will be created for them.

![invite member section](../../assets/screenshots/invite-member.png)

## Create a token for a Workspace

To interact with a Workspace's resources programmatically, you need to add an
API token for that Workspace. Your existing API tokens are displayed on
[the settings page](https://modal.com/settings/tokens) and new API tokens can be added for a
particular Workspace.

After adding a token for a Workspace to your Modal config file you can activate
that Workspace's profile using the CLI (see below).

As an manager or workspace owner you can manage active tokens for a workspace on
[the member tokens page](https://modal.com/settings/tokens/member-tokens). For more information on API
token management see the
[documentation about configuration](https://modal.com/docs/reference/modal.config).

## Switching active Workspace

When on the dashboard or using the CLI, the active profile determines which
personal or organizational Workspace is associated with your actions.

### Dashboard

You can switch between organization Workspaces and your Personal Workspace by
using the workspace selector at the top of [the dashboard](https://modal.com/home).

### CLI

To switch the Workspace associated with CLI commands, use
`modal profile activate`.

## Administrating workspace members

Workspaces have three different levels of access privileges:

- Owner
- Manager
- Member

The user that creates a workspace is automatically set as the **Owner** for that
workspace. The owner can assign any other roles within the workspace, as well as
remove other members of the workspace.

A **Manager** within a workspace can assign all roles except **Owner** and can
also remove other members of the workspace.

A **Member** of a workspace can not assign any access privileges within the
workspace but can otherwise perform any action like running and deploying apps
and modify Secrets.

As an Owner or Manager you can administrate the access privileges of other
members on the `Workspace Management` tab in [settings](https://modal.com/settings/workspace-management).

## Leaving a Workspace

To leave a workspace, navigate to [the settings page](https://modal.com/settings/workspaces) and
click "Leave" on a listed Workspace. There must be at least one owner assigned
to a workspace.

#### Environments

# Environments

Environments are sub-divisions of workspaces, allowing you to deploy the same app
(or set of apps) in multiple instances for different purposes without changing
your code. Typical use cases for environments include having one `dev`
environment and one `prod` environment, preventing overwriting production apps
when developing new features, while still being able to deploy changes to a
"live" and potentially complex structure of apps.

Each environment has its own set of [Secrets](https://modal.com/docs/guide/secrets) and any
object lookups performed from an app in an environment will by default look for
objects in the same environment.

By default, every workspace has a single Environment called "main". New
Environments can be created on the CLI:

```sh
modal environment create dev
```

(You can run `modal environment --help` for more info)

Once created, Environments show up as a dropdown menu in the navbar of the
[Modal dashboard](https://modal.com/home), letting you set browse all Modal Apps and Secrets
filtered by which Environment they were deployed to.

Most CLI commands also support an `--env` flag letting you specify which
Environment you intend to interact with, e.g.:

```sh
modal run --env=dev app.py
modal volume create --env=dev storage
```

To set a default Environment for your current CLI profile you can use
`modal config set-environment`, e.g.:

```sh
modal config set-environment dev
```

Alternatively, you can set the `MODAL_ENVIRONMENT` environment variable.

## Environment web suffixes

Environments have a 'web suffix' which is used to make
[web endpoint URLs](https://modal.com/docs/guide/webhook-urls) unique across your workspace. One
Environment is allowed to have no suffix (`""`).

## Cross environment lookups

It's possible to explicitly look up objects in Environments other than the Environment
your app runs within:

```python
production_secret = modal.Secret.from_name(
    "my-secret",
    environment_name="main"
)
```

```python notest
modal.Function.from_name(
    "my_app",
    "some_function",
    environment_name="dev"
)
```

However, the `environment_name` argument is optional and omitting it will use
the Environment from the object's associated App or calling context.

#### Modal user account setup

# Modal user account setup

To run and deploy applications on Modal you'll need to sign up and create a user
account.

You can visit the [signup](https://modal.com/signup) page to begin the process or execute
[`modal setup`](https://modal.com/docs/reference/cli/setup#modal-setup) on the command line.

Users can also be provisioned through [Okta SSO](https://modal.com/docs/guide/okta-sso), which is
an enterprise feature that you can request. For the typical user you'll sign-up
using an existing GitHub account. If you're interested in authenticating with
other identity providers let us know at <support@modal.com>.

## What GitHub permissions does signing up require?

- `user:email` — gives us the emails associated with the GitHub account.
- `read:org` (invites only) — needed for Modal workspace invites. Note: this
  only allows us to see what organization memberships you have
  ([GitHub docs](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/scopes-for-oauth-apps)).
  We won't be able to access any code repositories or other details.

## How can I change my email?

You can change your email on the [settings](https://modal.com/settings) page.

#### Service users

# Service Users (beta)

Service users are programmatic accounts that allow automated systems to interact with Modal. They're ideal for CI/CD pipelines, automated deployments, and other workflows that need to authenticate.

## Create a Service User

Service users are only available for shared workspaces. You will need workspace owner or manager privileges to create service users.

To create a service user:

1. Go to your workspace [tokens settings page](https://modal.com/settings/tokens/service-users)
2. Click **New Service User**
3. Enter a name for your service user (must be lowercase alphanumeric, can contain hyphens or underscores)
4. Click **Create**

After creation, you'll see the `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. **This is the only time you can view the token secret** for security reasons.

## Use Service User Tokens

Set the service user credentials as environment variables in your automated environment:

```bash
export MODAL_TOKEN_ID=your-token-id
export MODAL_TOKEN_SECRET=your-token-secret
```

Once configured, you can use Modal's CLI and Python SDK as usual:

```bash
modal deploy your_app.py
```

## Delete a Service User

To remove a service user:

1. Go to the [tokens settings page](https://modal.com/settings/tokens/service-users)
2. Find the service user in the table
3. Click **Delete** when you hover over the row

## Permissions

Service users have the same permissions as workspace members. They cannot do actions that are only permitted for a workspace owner or manager. To learn more about members, managers, and owners, see this [workspace](https://modal.com/docs/guide/workspaces#administrating-workspace-members) section.

### Other topics

#### Modal 1.0 migration guide

# Modal 1.0 migration guide

We released version 1.0 of the Modal Python SDK in May 2025.
This release signifies an increased commitment to API stability and implies
some changes to our development workflow.

Preceding the 1.0 release, we introduced a number of deprecations and changes
based on feedback that we received from early users. These changes were intended
to address pain points and reduce confusion about some aspects of the Modal API.
While adapting to them requires some changes to existing code, we believe that
they’ll make it easier to use Modal going forward.

This page highlights the major changes for 1.0 and provides some advice for how
to migrate your code to the new stable APIs. Most deprecations introduced prior
to the release of v1.0 will not be enforced (actually cause breaking changes)
until a subsequent minor (v1.x) release, but we recommend updating your code so
that you can take advantage of new features and avoid any future issues.

## Deprecating `Image.copy_*` methods

_Introduced in: v0.72.11_

We recently introduced new `Image` methods — `Image.add_local_dir` and
`Image.add_local_file` — to replace the existing `Image.copy_local_dir` and
`Image.copy_local_file`.

The new methods subsume the functionality of the old ones, but their default
behavior is different and more performant. By default, files will be mounted to
the container at runtime rather than copied into a new `Image` layer. This can
speed up development substantially when iterating on the contents of the files.

Building a new `Image` layer should be necessary only when subsequent build
steps will use the added files. In that case, you can pass `copy=True` in
`Image.add_local_file` or `Image.add_local_dir`.

The `Image.add_local_dir` method also has an `ignore=` parameter, which you can
use to pass file-matching patterns (using dockerignore rules) or predicate
functions to exclude files.

## Deprecating `Mount` as part of the public API

_Introduced in: v0.72.4_ | _Enforced in: v1.0.0_

Currently, local files can be mounted to the container filesystem either by
including them in the `Image` definition or by passing a `modal.Mount` object
directly to the `App.function` or `App.cls` decorators. As part of the 1.0
release, we are simplifying the container filesystem configuration to be defined
only by the `Image` used for each Function. This implies deprecation of the
following:

- The `mount=` parameter of `App.function` and `App.cls`
- The `context_mount=` parameter of several `modal.Image` methods
- The `Image.copy_mount` method
- The `Mount` object

Code that uses the `mount=` parameter of `App.function` and `App.cls` should be
migrated to pass those files / directories to the `Image` used by that Function
or Cls, i.e. using the `Image.add_local_file`, `Image.add_local_dir`, or
`Image.add_local_python_source` methods:

```python notest
# Mounting local files

# Old way (deprecated)
mount = modal.Mount.from_local_dir("data").add_local_file("config.yaml")
@app.function(image=image, mount=mount)
def f():
    ...

# New way
image = image.add_local_dir("data", "/root/data").add_local_file("config.yaml", "/root/config.yaml")
@app.function(image=image)
def f():
    ...

## Mounting local Python source code

# Old way (deprecated)
mount = modal.Mount.from_local_python_packages("my-lib"))
@app.function(image=image, mount=mount)
def f()
    ...

# New way
image = image.add_local_python_source("my-lib")
@app.function(image=image)
def f(...):
    ...

## Using Image.copy_mount

# Old way (deprecated)
mount = modal.Mount.from_local_dir("data").add_local_file("config.yaml")
image.copy_mount(mount)

# New way
image.add_local_dir("data", "root/data").add_local_file("config.yaml", "/root/config.yaml")
```

Code that uses the `context_mount=` parameter of `Image.from_dockerfile` and
`Image.dockerfile_commands` methods can delete that parameter; we now
automatically infer the files that need to be included in the context.

## Deprecating the `@modal.build` decorator

_Introduced in: v0.72.17_

As part of consolidating the filesystem configuration API, we are also
deprecating the `modal.build` decorator.

For use cases where `modal.build` would previously have been the suggested
approach (e.g., downloading model weights or other large assets to the
container filesystem), we now recommend using a `modal.Volume` instead. The
main advantage of storing weights in a `Volume` instead of an `Image` is that
the weights do not need to be re-downloaded every time you change something else
about the `Image` definition.

Many frameworks, such as Hugging Face, automatically cache downloaded model
weights. When using these frameworks, you just need to ensure that you mount a
`modal.Volume` to the expected location of the framework’s cache:

```python notest
cache_vol = modal.Volume.from_name("hf-hub-cache")
@app.cls(
    image=image.env({"HF_HUB_CACHE": "/cache"}),
    volumes={"/cache": cache_vol},
    ...
)
class Model:
    @modal.enter()
    def load_model(self):
        self.model = ModelClass.from_pretrained(...)
```

For frameworks that don’t support automatic caching, you could write a separate
function to download the weights and write them directly to the Volume, then
`modal run` against this function before you deploy.

In some cases (e.g., if the step runs very quickly), you may wish for the logic
currently decorated with `@modal.build` to continue modifying the Image
filesystem. In that case, you can extract the method as a standalone function
and pass it to `Image.run_function`:

```python notest
def download_weights():
    ...

image = image.run_function(download_weights)
```

## Requiring explicit inclusion of local Python dependencies

_Introduced in: 0.73.11_ | _Enforced in: 1.0.0_

Prior to 1.0, Modal will inspect the modules that are imported when running
your App code and automatically include any "local" modules in the remote
container environment. This behavior is referred to as "automounting".

While convenient, this approach has a number of edge cases and surprising
behaviors, such as ignoring modules with imports that are deferred using
`Image.imports`. Additionally, it is difficult to configure the automounting
behavior to, e.g., ignore large data files that are stored within your local
Python project directories.

Going forward, it will be necessary to explicitly include the local dependencies
of your Modal App. The easiest way to do this is with
[`Image.add_local_python_source`](https://modal.com/docs/reference/modal.Image#add_local_python_source):

```python notest
import modal
import helpers

image = modal.Image.debian_slim().add_local_python_source("helpers")
```

In the period leading up to the change in default behavior, the Modal client
will issue deprecation warnings when automounted modules are not included
in the Image. Updating the Image definition will remove these warnings.

Note that Modal will continue to automatically include the source module or
package defining the App itself. We're introducing a new App or Function-level
parameter, `include_source`, which can be set to `False` in cases where this is
not desired (i.e., because your Image definition already includes the App
source).

## Renaming autoscaler parameters

_Introduced in: v0.73.76_

We're renaming several parameters that configure autoscaling behavior:

- `keep_warm` is now `min_containers`
- `concurrency_limit` is now `max_containers`
- `container_idle_timeout` is now `scaledown_window`

The renaming is intended to address some persistent confusion about
the meaning of these parameters. The migration path is a simple
find-and-replace operation.

Additionally, we're promoting a fourth parameter, `buffer_containers`,
from experimental status (previously `_experimental_buffer_containers`).
Like `min_containers`, `buffer_containers` can help mitigate cold-start
penalties by overprovisioning containers while the Function is active.

## Renaming `modal.web_endpoint` to `modal.fastapi_endpoint`

_Introduced in: v0.73.89_

We're renaming the `modal.web_endpoint` decorator to `modal.fastapi_endpoint`
so that the implicit dependency on FastAPI is more clear. This can be a
simple name substitution in your code as the semantics are otherwise identical.

We may reintroduce a lightweight `modal.web_endpoint` without external
dependencies in the future.

## Replacing `allow_concurrent_inputs` with `@modal.concurrent`

_Introduced in: v0.73.148_

The `allow_concurrent_inputs` parameter is being replaced with a new decorator,
`@modal.concurrent`. The decorator can be applied either to a Function or a Cls.
We're moving the input concurrency feature out of "Beta" status as part of this
change.

The new decorator exposes two distinct parameters: `max_inputs` (the limit
on the number of inputs the Function will concurrently accept) and
`target_inputs` (the level of concurrency targeted by the Modal autoscaler).
The simplest migration path is to replace `allow_concurrent_inputs=N` with
`@modal.concurrent(max_inputs=N)`:

```python notest
# Old way, with a function (deprecated)
@app.function(allow_concurrent_inputs=1000)
def f(...):
    ...

# New way, with a function
@app.function()
@modal.concurrent(max_inputs=1000)
def f(...):
    ...

# Old way, with a class (deprecated)
@app.cls(allow_concurrent_inputs=1000)
class MyCls:
    ...

# New way, with a class
@app.cls()
@modal.concurrent(max_inputs=1000)
class MyCls:
    ...
```

Setting `target_inputs` along with `max_inputs` may benefit performance by
reducing latency during periods where the container pool is scaling up. See the
[input concurrency guide](https://modal.com/docs/guide/concurrent-inputs) for more information.

## Deprecating the `.lookup` method on Modal objects

_Introduced in: v0.72.56_

Most Modal objects can be instantiated through two distinct methods:
`.from_name` and `.lookup`. The redundancy between these methods is a persistent
source of confusion.

The `.from_name` method is lazy: it operates entirely locally and instantiates
only a shell for the object. The local object won’t be associated with its
identity on the Modal server until you interact with it. In contrast, the
`.lookup` method is eager: it triggers a remote call to the Modal server, and it
returns a fully-hydrated object.

Because Modal objects can now be hydrated on-demand, when they are first
used, there is rarely any need to eagerly hydrate. Therefore, we’re deprecating
`.lookup` so that there’s only one obvious way to instantiate objects.

In most cases, the migration is a simple find-and-replace of `.lookup` →
`.from_name`.

One exception is when your code needs to access object metadata, such as its ID,
or a web endpoint's URL. In that case, you can explicitly force hydration of the
object by calling its `.hydrate()` method. There may be other subtle consequences,
such as errors being rasied at a different location if no object exists with the
given name.

## Removing support for custom Cls constructors

_Introduced in: v0.74.0_

Classes decorated with `App.cls` are no longer allowed to have a custom constructor
(`__init__` method). Instead, class parameterization should be exposed using
dataclass-style [`modal.parameter`](https://modal.com/docs/reference/modal.parameter) annotations:

```python notest
# Old way (deprecated)
@app.cls()
class MyCls:
    def __init__(self, name: str = "Bert"):
        self.name = name

# New way
@app.cls()
class MyCls:
    name: str = modal.parameter(default="Bert")
```

Modal will provide a synthetic constructor for classes that use `modal.parameter`.
Arguments to the synthetic constructor must be passed using keywords, so you may
need to update your calling code as well:

```python notest
obj = MyCls(name="Bert")  # name= is now required
```

We're making this change to address some persistent confusion about when
constructors execute for remote calls and what operations are allowed to run in
them. If your custom constructor performs any setup logic beyond storing the
parameter values, you should move it to a method decorated with
`@modal.enter()`.

Additionally, we're reducing the types that we support as class parameters to
a small number of primitives (`str`, `int`, `bool`, and `bytes`).

Limiting class parameterization to primitive types will also allow us to provide
better observability over parameterized class instances in the web dashboard,
CLI, and other contexts where it is not possible to represent arbitrary Python
objects.

If you need to parameterize classes across more complex types, you can implement
your own serialization logic, e.g. using strings as the wire format:

```python notest
@app.cls()
class MyCls:
    param_str: str = modal.parameter()

    @modal.enter()
    def deserialize_parameters(self):
        self.param_obj = SomeComplexType.from_str(self.param_str)
```

We recommend adopting interpretable constructor arguments (i.e., prefer
meaningful strings over pickled bytes) so that you will be able to get the most
benefit from future improvements to parameterized class observability.

## Simplifying Cls lookup patterns

_Introduced in: v0.73.26_

Modal previously supported several different patterns for looking up a `modal.Cls`
and remotely invoking one of its methods:

```python notest
# Documented pattern
MyCls = modal.Cls.from_name("my-app", "MyCls")
obj = MyCls()
obj.some_method.remote(...)

# Alternate pattern: skipping the object instantiation
MyCls = modal.Cls.from_name("my-app", "MyCls")
MyCls.some_method.remote(...)

# Alternate pattern: looking up the method as a Function
f = modal.Function.lookup("my-app", "MyCls.some_method")
f.remote(...)
```

While each pattern could successfully trigger a remote function call, there were
a number of subtle differences in behavior between them.

Going forward, we will only support the first pattern. Making remote calls to a
method on a deployed Cls will require you to (a) look up the object using
`modal.Cls` and (b) instantiate the object before calling its methods.

## Deprecating `modal.gpu` objects

_Introduced in: v0.73.31_

The `modal.gpu` objects are being deprecated; going forward, all GPU resource
configuration should be accomplished using strings.

This should be an easy code substitution, e.g. `gpu=modal.gpu.H100()` can be
replaced with `gpu="H100"`. When using the `count=` parameter of the GPU class,
simply append it to the name with a colon (e.g. `gpu="H100:8"`). In the case of
the `modal.gpu.A100(size="80GB")` variant, the name of the corresponding gpu is
`"A100-80GB"`.

Note that string arguments are case-insensitive, so `"H100"` and `"h100"` are
both accepted.

The main rationale for this change is that it will allow us to introduce new
GPU models in the future without requring users to upgrade their SDK.

## Requiring explicit invocation for module mode

_Introduced in: 0.73.58_

The Modal CLI allows you to reference the source code for your App as either
a file path (e.g. `src/my_app.py`) or as a module name (e.g. `src.my_app`).

As in Python, the choice has some implications for how relative imports are
resolved. To make this more salient, Modal will mirror Python going forwared
and require that you explicitly invoke module mode by passing `-m` on your
command line (e.g., `modal deploy -m src.my_app`).

#### File and project structure

# Project structure

## Apps spanning multiple files

When your project spans multiple files, more care is required to package the
full structure for running or deploying on Modal.

There are two main considerations: (1) ensuring that all of your Functions get
registered to the App, and (2) ensuring that any local dependencies get included
in the Modal container.

Say that you have a simple project that's distributed across three files:

```
src/
├── app.py  # Defines the `modal.App` as a variable named `app`
├── llm.py  # Imports `app` and decorates some functions
└── web.py  # Imports `app` and decorates other functions
```

With this structure, if you deploy using `modal deploy src/app.py`, Modal won't
discover the Functions defined in the other two modules, because they never get
imported.

If you instead run `modal deploy src/llm.py`, Modal will deploy the App with
just the Functions defined in that module.

One option would be to ensure that one module in the project transitively
imports all of the other modules and to point the `modal deploy` CLI at it, but
this approach can lead to an awkard project structure.

### Defining your project as a Python package

A better approach would be to define your project as a Python _package_ and to
use the Modal CLI's "module mode" invocation pattern.

In Python, a package is a directory containing an `__init__.py` file (and
usually some other Python modules). If you have a `src/__init__.py` that
imports all of the member modules, it will ensure that any decorated Functions
contained within them get registered to the App:

```python notest
# Contents of __init__.py
import .app
import .llm
import .web
```

_Important: use *relative* imports (`import .app`) between member modules._

Unfortunately, it's not enough just to set this up and make your deploy command
`modal deploy src/app.py`. Instead, you need to invoke Modal in _module mode_:
`modal deploy -m src.app`. Note the use of the `-m` flag and the module path
(`src.app` instead of `src/app.py`). Akin to `python -m ...`, this incantation
treats the target as a package rather than just a single script.

### App composition

As your project grows in scope, it may become helpful to organize it into
multiple component Apps, rather than having the project defined as one large
monolith. That way, as you iterate during development, you can target a specific
component, which will build faster and avoid any conflicts with concurrent work
on other parts of the project.

Projects set up this way can still be deployed as one unit by using `App.include`.
Say our project from above defines separate Apps in `llm.py` and `web.py` and then
adds a new `deploy.py` file:

```python notest
# Contents of deploy.py
import modal

from .llm import llm_app
from .web import web_app

app = modal.App("full-app").include(llm_app).include(web_app)
```

This lets you run `modal deploy -m src.deploy` to package everything in one
step.

**Note:** Since the multi-file app still has a single namespace for all
functions, it's important to name your Modal functions uniquely across the
project even when splitting it up across files: otherwise you risk some
functions "shadowing" others with the same name.

## Including local dependencies

Another factor to consider is whether Modal will package all of the local
dependencies that your App requires.

Even if your Modal App itself can be contained to a single file, any local
modules that file imports (like, say, a `helpers.py`) also need to be available
in the Modal container.

By default, Modal will automatically include the module or package where a
Function is defined in all containers that run that Function. So if the project
is set up as a package and the helper modules are part of that package, you
should be all set. If you're not using a package setup, or if the local
dependencies are external to your project's package, you'll need to explicitly
include them in the Image, i.e. with `modal.Image.add_local_python_source`.

**Note:** This behavior changed in Modal 1.0. Previously, Modal would
"automount" any local dependencies that were imported by your App source into a
container. This was changed to be more selective to avoid unnecessary inclusion
of large local packages.

#### Developing and debugging

# Developing and debugging

Modal makes it easy to run apps in the cloud, try code changes in the cloud, and
debug remotely executing code as if it were right there on your laptop. To speed
boost your inner dev loop, this guide provides a rundown of tools and techniques
for developing and debugging software in Modal.

## Interactivity

You can launch a Modal App interactively and have it drop you right into the
middle of the action, at an interesting callsite or the site of a runtime
detonation.

### Interactive functions

It is possible to start the interactive Python debugger or start an `IPython`
REPL right in the middle of your Modal App.

To do so, you first need to run your App in "interactive" mode by using the
`--interactive` / `-i` flag. In interactive mode, you can establish a connection
to the calling terminal by calling `interact()` from within your function.

For a simple example, you can accept user input with the built-in Python `input`
function:

```python
@app.function()
def my_fn(hidden):
    modal.interact()

    x = input("Enter a number: ")
    if hidden == x:
        print(f"Your number is {x}, which is the hidden value!")
    else:
        print(f"Your number is {x}, which is not the hidden value")
```

Now when you run your app with the `--interactive` flag, you're able to send
inputs to your app, even though it's running in a remote container!

```shell
modal run -i guess_number.py --hidden 5
Enter a number: 5
Your number is 5, which is the hidden value!
```

For a more interesting example, you can [`pip_install("ipython")`](https://modal.com/docs/reference/modal.Image#pip_install)
and start an `IPython` REPL dynamically anywhere in your code:

```python
@app.function()
def f():
    model = expensive_function()
    # play around with model
    modal.interact()
    import IPython
    IPython.embed()
```

The built-in Python debugger can be initiated with the language's `breakpoint()`
function. For convenience, breakpoints call `interact` automatically.

```python
@app.function()
def f():
    x = "10point3"
    breakpoint()
    answer = float(x)
```

### Debugging Running Containers

#### Debug Shells

Modal also lets you run interactive commands on your running Containers from the
terminal -- much like `ssh`-ing into a traditional machine or cloud VM.

To run a command inside a running Container, you first need to get the Container
ID. You can view all running Containers and their Container IDs with
[`modal container list`](https://modal.com/docs/reference/cli/container).

After you obtain the Container ID, you can connect to the Container with `modal shell [container-id]`. This launches a "Debug Shell" that comes with some preinstalled tools:

- `vim`
- `nano`
- `ps`
- `strace`
- `curl`
- `py-spy`
- and more!

You can use a debug shell to examine or terminate running processes, modify the Container filesystem, run commands, and more. You can also install additional packages using your Container's package manager (ex. `apt`).

Note that debug shells will terminate immediately once your Container has finished running.

#### `modal container exec`

You can also execute a specific command in a running Container with `modal container exec [container-id] [command...]`. For example, to see what files are in `/root`, you can run `modal container exec [container-id] ls /root`.

```
❯ modal container list
                         Active Containers in environment: nathan-dev
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Container ID                  ┃ App ID                    ┃ App Name ┃ Start Time           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ ta-01JK47GVDMWMGPH8MQ0EW30Y25 │ ap-FSuhQ4LpvNAt5b6mKi1CDw │ my-app   │ 2025-02-02 16:02 EST │
└───────────────────────────────┴───────────────────────────┴──────────┴──────────────────────┘

❯ modal container exec ta-01JK47GVDMWMGPH8MQ0EW30Y25 ls /root
__pycache__  test00.py
```

Note that your executed command will terminate immediately once your Container
has finished running.

By default, commands will be run within a
[pseudoterminal (PTY)](https://en.wikipedia.org/wiki/Pseudoterminal), but this
can be disabled with the `--no-pty` flag.

#### Live container profiling

When a container or input is seemingly stuck or not making progress,
you can use the Modal web dashboard to find out what code that's executing in the
container in real time. To do so, look for **Live Profiling** in the **Containers** tab in your
function dashboard.

![Live container profiling](https://modal-public-assets.s3.us-east-1.amazonaws.com/live-profiling-bigger.gif)

### Debugging Container Images

You can also launch an interactive shell in a new Container with the same
environment as your Function. This is handy for debugging issues with your
Image, interactively refining build commands, and exploring the contents of
[`Volume`](https://modal.com/docs/reference/modal.Volume)s and
[`NetworkFileSystem`](https://modal.com/docs/reference/modal.NetworkFileSystem)s.

The primary interface for accessing this feature is the
[`modal shell`](https://modal.com/docs/reference/cli/shell) CLI command, which accepts a Function
name in your App (or prompts you to select one, if none is provided), and runs
an interactive command on the same image as the Function, with the same
[`Secret`](https://modal.com/docs/reference/modal.Secret)s and
[`NetworkFileSystem`](https://modal.com/docs/reference/modal.NetworkFileSystem)s attached as the selected Function.

The default command is `/bin/bash`, but you can override this with any other
command of your choice using the `--cmd` flag.

Note that `modal shell [filename].py` does not attach a shell to a running Container of the
Function, but instead creates a fresh instance of the underlying Image. To attach a shell to a running Container, use `modal shell [container-id]` instead.

## Live updating

### Hot reloading with `modal serve`

Modal has the command `modal serve <filename.py>`, which creates a loop that
live updates an App when any of the supporting files change.

Live updating works with web endpoints, syncing your changes as you make them,
and it also works well with cron schedules and job queues.

```python
import modal

app = modal.App(image=modal.Image.debian_slim().pip_install("fastapi"))

@app.function()
@modal.fastapi_endpoint()
def f():
    return "I update on file edit!"

@app.function(schedule=modal.Period(seconds=5))
def run_me():
    print("I also update on file edit!")
```

If you edit this file, the `modal serve` command will detect the change and
update the code, without having to restart the command.

## Observability

Each running Modal App, including all ephemeral Apps, streams logs and resource
metrics back to you for viewing.

On start, an App will log a dashboard link that will take you its App page.

```shell
$ python3 main.py
✓ Initialized. View app page at https://modal.com/apps/ap-XYZ1234.
...
```

From this page you can access the following:

- logs, both from your application and system-level logs from Modal
- compute resource metrics (CPU, RAM, GPU)
- function call history, including historical success/failure counts

### Debug logs

You can enable Modal's client debug logs by setting the `MODAL_LOGLEVEL` environment variable to `DEBUG`.
Running the following will show debug logging from the Modal client running locally.

```bash
MODAL_LOGLEVEL=DEBUG modal run hello.py
```

To enable debug logs in the Modal client running in the remote container, you can set `MODAL_LOGLEVEL` using
a Modal [`Secret`](https://modal.com/docs/reference/modal.Secret).

```python
@app.function(secrets=[modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"})])
def f():
    print("Hello, world!")
```

### Client tracebacks

To see a traceback (a.k.a [stack trace](https://en.wikipedia.org/wiki/Stack_trace)) for a client-side exception, you can set the `MODAL_TRACEBACK` environment variable to `1`.

```bash
MODAL_TRACEBACK=1 modal run my_app.py
```

We encourage you to report cases where you need to enable this functionality, as it's indication of an issue in Modal.

#### Developing Modal code with LLMs

# Developing Modal code with LLMs

Excellent developer experience is at the core of Modal. This also means that Modal works well with code generation agents, especially those that can run CLI commands like `modal run` in an implement, test and debug loop, like Amp, Claude Code, Cursor's agent mode, Gemini CLI, etc.

There are of course also many concepts and design patterns that are unique to Modal, so below we gather rules and guidelines that we have found useful when developing Modal code with LLMs. You can paste/import this into your `AGENTS.md`, `CLAUDE.md`, `.cursor/rules/modal.mdc`, etc. or use it as a starting point for your own rules or prompts.

````markdown
# Modal Rules and Guidelines for LLMs

This file provides rules and guidelines for LLMs when implementing Modal code.

## General

- Modal is a serverless cloud platform for running Python code with minimal configuration
- Designed for AI/ML workloads but supports general-purpose cloud compute
- Serverless billing model - you only pay for resources used

## Modal documentation

- Extensive documentation is available at: modal.com/docs (and in markdown format at modal.com/llms-full.txt)
- A large collection of examples is available at: modal.com/docs/examples (and github.com/modal-labs/modal-examples)
- Reference documentation is available at: modal.com/docs/reference

Always refer to documentation and examples for up-to-date functionality and exact syntax.

## Core Modal concepts

### App

- A group of functions, classes and sandboxes that are deployed together.

### Function

- The basic unit of serverless execution on Modal.
- Each Function executes in its own container, and you can configure different Images for different Functions within the same App:

  ```python
  image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers")
    .apt_install("ffmpeg")
    .run_commands("mkdir -p /models")
  )

  @app.function(image=image)
  def square(x: int) -> int:
    return x * x
  ```

- You can configure individual hardware requirements (CPU, memory, GPUs, etc.) for each Function.

  ```python
  @app.function(
    gpu="H100",
    memory=4096,
    cpu=2,
  )
  def inference():
    ...
  ```

  Some examples specificly for GPUs:

  ```python
  @app.function(gpu="A10G")  # Single GPU, e.g. T4, A10G, A100, H100, or "any"
  @app.function(gpu="A100:2")  # Multiple GPUs, e.g. 2x A100 GPUs
  @app.function(gpu=["H100", "A100", "any"]) # GPU with fallbacks
  ```

- Functions can be invoked in a number of ways. Some of the most common are:
  - `foo.remote()` - Run the Function in a separate container in the cloud. This is by far the most common.
  - `foo.local()` - Run the Function in the same context as the caller. Note: This does not necessarily mean locally on your machine.
  - `foo.map()` - Parallel map over a set of inputs.
  - `foo.spawn()` - Calls the function with the given arguments, without waiting for the results. Terminating the App will also terminate spawned functions.
- Web endpoint: You can turn any Function into an HTTP web endpoint served by adding a decorator:

  ```python
  @app.function()
  @modal.fastapi_endpoint()
  def fastapi_endpoint():
    return {"status": "ok"}

  @app.function()
  @modal.asgi_app()
  def asgi_app():
    app = FastAPI()
    ...
    return app
  ```

- You can run Functions on a schedule using e.g. `@app.function(schedule=modal.Period(minutes=5))` or `@app.function(schedule=modal.Cron("0 9 * * *"))`.

### Classes (a.k.a. `Cls`)

- For stateful operations with startup/shutdown lifecycle hooks. Example:

  ```python
  @app.cls(gpu="A100")
  class ModelServer:
      @modal.enter()
      def load_model(self):
          # Runs once when container starts
          self.model = load_model()

      @modal.method()
      def predict(self, text: str) -> str:
          return self.model.generate(text)

      @modal.exit()
      def cleanup(self):
          # Runs when container stops
          cleanup()
  ```

### Other important concepts

- Image: Represents a container image that Functions can run in.
- Sandbox: Allows defining containers at runtime and securely running arbitrary code inside them.
- Volume: Provide a high-performance distributed file system for your Modal applications.
- Secret: Enables securely providing credentials and other sensitive information to your Modal Functions.
- Dict: Distributed key/value store, managed by Modal.
- Queue: Distributed, FIFO queue, managed by Modal.

## Differences from standard Python development

- Modal always executes code in the cloud, even while you are developing. You can use Environments for separating development and production deployments.
- Dependencies: It's common and encouraged to have different dependency requirements for different Functions within the same App. Consider defining dependencies in Image definitions (see Image docs) that are attached to Functions, rather than in global `requirements.txt`/`pyproject.toml` files, and putting `import` statements inside the Function `def`. Any code in the global scope needs to be executable in all environments where that App source will be used (locally, and any of the Images the App uses).

## Modal coding style

- Modal Apps, Volumes, and Secrets should be named using kebab-case.
- Always use `import modal`, and qualified names like `modal.App()`, `modal.Image.debian_slim()`.
- Modal evolves quickly, and prints helpful deprecation warnings when you `modal run` an App that uses deprecated features. When writing new code, never use deprecated features.

## Common commands

Running `modal --help` gives you a list of all available commands. All commands also support `--help` for more details.

### Running your Modal app during development

- `modal run path/to/your/app.py` - Run your app on Modal.
- `modal run -m module.path.to.app` - Run your app on Modal, using the Python module path.
- `modal serve modal_server.py` - Run web endpoint(s) associated with a Modal app, and hot-reload code on changes. Will print a URL to the web endpoint(s). Note: you need to use `Ctrl+C` to interrupt `modal serve`.

### Deploying your Modal app

- `modal deploy path/to/your/app.py` - Deploy your app (Functions, web endpoints, etc.) to Modal.
- `modal deploy -m module.path.to.app` - Deploy your app to Modal, using the Python module path.

Logs:

- `modal app logs <app_name>` - Stream logs for a deployed app. Note: you need to use `Ctrl+C` to interrupt the stream.

### Resource management

- There are CLI commands for interacting with resources like `modal app list`, `modal volume list`, and similarly for `secret`, `dict`, `queue`, etc.
- These also support other command than `list` - use e.g. `modal app --help` for more.

## Testing and debugging

- When using `app.deploy()`, you can wrap it in a `with modal.enable_output():` block to get more output.
````

#### Jupyter notebooks

# Jupyter notebooks

> **Note:** This page is about running Jupyter on Modal. For our hosted notebooks product with real-time collaboration, see [Modal Notebooks](https://modal.com/docs/guide/notebooks-modal).

You can use the Modal client library in notebook environments like Jupyter! Just
`import modal` and use as normal. You will likely need to use [`app.run`](https://modal.com/docs/guide/apps#ephemeral-apps) to create an ephemeral app to run your functions:

```python,notest
# Cell 1

import modal

app = modal.App()

@app.function()
def my_function(x):
    ...

# Cell 2

with modal.enable_output():
    with app.run():
        my_function.remote(42)
```

## Known issues

- **Interactive shell and interactive functions are not supported.**

  These can only be run within a live terminal session, so they are not
  supported in notebooks.

- **Local and remote Python versions must match.**

  When defining Modal Functions in a Jupyter notebook, the Function automatically
  has `serialized=True` set. This implies that the versions of Python and any third-
  party libraries used in your Modal container must match the version you have locally,
  so that the function can be deserialized remotely without errors.

If you encounter issues not documented above, try restarting the notebook kernel, as it may be
in a broken state, which is common in notebook development.

If the issue persists, contact us [in our Slack](https://modal.com/slack).

We are working on removing these known issues so that writing Modal applications
in a notebook feels just like developing in regular Python modules and scripts.

## Jupyter inside Modal

You can run Jupyter in Modal using the `modal launch` command. For example:

```
$ modal launch jupyter --gpu a10g
```

That will start a Jupyter instance with an A10G GPU attached. You'll be able to
access the app with via a
[Modal Tunnel URL](https://modal.com/docs/guide/tunnels#tunnels-beta). Jupyter
will stop running whenever you stop Modal call in your terminal.

See `--help` for additional options.

## Further examples

- [Basic demonstration of running Modal in a notebook](https://github.com/modal-labs/modal-examples/blob/main/11_notebooks/basic.ipynb)
- [Running Jupyter server within a Modal function](https://github.com/modal-labs/modal-examples/blob/main/11_notebooks/jupyter_inside_modal.py)

#### Asynchronous API usage

# Asynchronous API usage

All of the functions in Modal are available in both standard (blocking) and
asynchronous variants. The async interface can be accessed by appending `.aio`
to any function in the Modal API.

For example, instead of `my_modal_function.remote("hello")` in a blocking
context, you can use `await my_modal_function.remote.aio("hello")` to get an
asynchronous coroutine response, for use with Python's `asyncio` library.

```python
import asyncio
import modal

app = modal.App()

@app.function()
async def myfunc():
    ...

@app.local_entrypoint()
async def main():
    # execute 100 remote calls to myfunc in parallel
    await asyncio.gather(*[myfunc.remote.aio() for i in range(100)])
```

This is an advanced feature. If you are comfortable with asynchronous
programming, you can use this to create arbitrary parallel execution patterns,
with the added benefit that any Modal functions will be executed remotely.

## Async functions

Regardless if you use an async runtime (like `asyncio`) in your usage of _Modal
itself_, you are free to define your `app.function`-decorated function bodies
as either async or blocking. Both kinds of definitions will work for remote
Modal function calls from both any context.

An async function can call a blocking function, and vice versa.

```python
@app.function()
def blocking_function():
    return 42

@app.function()
async def async_function():
    x = await blocking_function.remote.aio()
    return x * 10

@app.local_entrypoint()
def blocking_main():
    print(async_function.remote())  # => 420
```

If a function is configured to support multiple concurrent inputs per container,
the behavior varies slightly between blocking and async contexts:

- In a blocking context, concurrent inputs will run on separate Python threads.
  These are subject to the GIL, but they can still lead to race conditions if
  used with non-threadsafe objects.
- In an async context, concurrent inputs are simply scheduled as coroutines on
  the executor thread. Everything remains single-threaded.

#### Global variables

# Global variables

There are cases where you might want objects or data available in **global**
scope. For example:

- You need to use the data in a scheduled function (scheduled functions don't
  accept arguments)
- You need to construct objects (e.g. Secrets) in global scope to use as
  function annotations
- You don't want to clutter many function signatures with some common arguments
  they all use, and pass the same arguments through many layers of function
  calls.

For these cases, you can use the `modal.is_local` function, which returns `True`
if the app is running locally (initializing) or `False` if the app is executing
in the cloud.

For instance, to create a [`modal.Secret`](https://modal.com/docs/guide/secrets) that you can pass
to your function decorators to create environment variables, you can run:

```python
import os

if modal.is_local():
    pg_password = modal.Secret.from_dict({"PGPASS": os.environ["MY_LOCAL_PASSWORD"]})
else:
    pg_password = modal.Secret.from_dict({})

@app.function(secrets=[pg_password])
def get_secret_data():
    connection = psycopg2.connect(password=os.environ["PGPASS"])
    ...
```

## Warning about regular module globals

If you try to construct a global in module scope using some local data _without_
using something like `modal.is_local`, it might have unexpected effects since
your Python modules will be not only be loaded on your local machine, but also
on the remote worker.

E.g., this will typically not work:

```python notest
# blob.json doesn't exist on the remote worker, so this will cause an error there
data_blob = open("blob.json", "r").read()

@app.function()
def foo():
    print(data_blob)
```

#### Region selection

# Region selection

Modal allows you to specify which cloud region you would like to run a Function in. This may be useful if:

- you are required (for regulatory reasons or by your customers) to process data within certain regions.
- you want to reduce egress fees that result from reading data from a dependency like S3.
- you have a latency-sensitive app where app endpoints need to run near an external DB.

Note that regardless of what region your Function runs in, all Function inputs and outputs go through Modal's control plane in us-east-1.

## Pricing

A multiplier on top of our [base usage pricing](https://modal.com/pricing) will be applied to any function that has a cloud region defined.

| **Region**             | **Multiplier** |
| ---------------------- | -------------- |
| Any region in US/EU/AP | 1.25x          |
| All other regions      | 2.5x           |

Here's an example: let's say you have a function that uses 1 T4, 1 CPU core, and 1GB memory. You've specified that the function should run in `us-east-2`. The cost to run this function for 1 hour would be `((T4 hourly cost) + (CPU hourly cost for one core) + (Memory hourly cost for one GB)) * 1.25`.

If you specify multiple regions and they span the two categories above, we will apply the smaller of the two multipliers.

## Specifying a region

To run your Modal Function in a specific region, pass a `region=` argument to the `function` decorator.

```python
import os
import modal

app = modal.App("...")

@app.function(region="us-east") # also supports a list of options, for example region=["us-central", "us-east"]
def f():
    print(f"running in {os.environ['MODAL_REGION']}") # us-east-1, us-east-2, us-ashburn-1, etc.
```

You can specify a region in addition to the underlying cloud, `@app.function(cloud="aws", region="us-east")` would run your Function only in `"us-east-1"` or `"us-east-2"` for instance.

## Region options

Modal offers varying levels of granularity for regions. Use broader regions when possible, as this increases the pool of available resources your Function can be assigned to, which improves cold-start time and availability.

### United States ("us")

Use `region="us"` to select any region in the United States.

<!-- TODO: auto-generate this table, this is not sustainable -->

```
     Broad            Specific             Description
 ==============================================================
  "us-east"           "us-east-1"          AWS Virginia
                      "us-east-2"          AWS Ohio
                      "us-east1"           GCP South Carolina
                      "us-east4"           GCP Virginia
                      "us-east5"           GCP Ohio
                      "us-ashburn-1"       OCI Virginia
                      "eastus"             AZR Virginia
                      "eastus2"            AZR Virginia
 --------------------------------------------------------------
  "us-central"        "us-central1"        GCP Iowa
                      "us-chicago-1"       OCI Chicago
                      "us-phoenix-1"       OCI Phoenix
                      "centralus"          AZR Iowa
                      "northcentralus"     AZR Illinois
                      "southcentralus"     AZR Texas
                      "westcentralus"      AZR Wyoming
 --------------------------------------------------------------
  "us-west"           "us-west-1"          AWS California
                      "us-west-2"          AWS Oregon
                      "us-west1"           GCP Oregon
                      "us-west3"           GCP Utah
                      "us-west4"           GCP Nevada
                      "us-sanjose-1"       OCI San Jose
                      "westus"             AZR California
                      "westus2"            AZR Washington
                      "westus3"            AZR Phoenix
```

### Europe ("eu")

Use `region="eu"` to select any region in Europe.

```
     Broad            Specific             Description
 ==============================================================
  "eu-west"           "eu-central-1"       AWS Frankfurt
                      "eu-west-1"          AWS Ireland
                      "eu-west-3"          AWS Paris
                      "europe-west1"       GCP Belgium
                      "europe-west3"       GCP Frankfurt
                      "europe-west4"       GCP Netherlands
                      "eu-frankfurt-1"     OCI Frankfurt
                      "eu-paris-1"         OCI Paris
                      "francecentral"      AZR Paris
                      "germanywestcentral" AZR Frankfurt
                      "switzerlandnorth"   AZR Zurich
                      "westeurope"         AZR Netherlands
 --------------------------------------------------------------
  "eu-north"          "eu-north-1"         AWS Stockholm
                      "swedencentral"      AZR Gävle
                      "northeurope"        AZR Ireland
                      "norwayeast"         AZR Norway
```

### Asia–Pacific ("ap")

Use `region="ap"` to select any region in Asia–Pacific.

```
     Broad            Specific             Description
 ==============================================================
  "ap-northeast"      "asia-northeast3"    GCP Seoul
                      "asia-northeast1"    GCP Tokyo
                      "ap-northeast-1"     AWS Tokyo
                      "ap-northeast-3"     AWS Osaka
                      "koreacentral"       AZR Seoul
                      "japaneast"          AZR Tokyo, Saitama
                      "japanwest"          AZR Osaka
 --------------------------------------------------------------
  "ap-southeast"      "asia-southeast1"    GCP Singapore
                      "ap-southeast-3"     AWS Jakarta
                      "southeastasia"      AZR Singapore
 --------------------------------------------------------------
  "ap-south"          "ap-south-1"         AWS Mumbai
                      "centralindia"       AZR Pune
                      "westindia"          AZR Mumbai
```

### Other regions

```
     Broad            Specific             Description
 ==============================================================
  "ca"                "ca-central-1"       AWS Montreal
                      "ca-toronto-1"       OCI Toronto
                      "canadacentral"      AZR Toronto
                      "canadaeast"         AZR Quebec
 --------------------------------------------------------------
  "uk"                "uk-london-1"        OCI London
                      "europe-west2"       GCP London
                      "eu-west-2"          AWS London
                      "uksouth"            AZR London
 --------------------------------------------------------------
  "jp"                "ap-northeast-1"     AWS Tokyo
                      "ap-northeast-3"     AWS Osaka
                      "asia-northeast1"    GCP Tokyo
                      "japaneast"          AZR Tokyo, Saitama
                      "japanwest"          AZR Osaka
 --------------------------------------------------------------
  "me"                "me-west1"           GCP Tel Aviv
                      "uaenorth"           AZR Dubai
 --------------------------------------------------------------
  "sa"                "sa-east-1"          AWS São Paulo
                      "brazilsouth"        AZR São Paulo State
 --------------------------------------------------------------
  "au"                "ap-melbourne-1"     OCI Melbourne
                      "ap-sydney-1"        OCI Sydney
                      "australiaeast"      AZR New South Wales
 --------------------------------------------------------------
  "af"                "southafricanorth"   AZR Johannesburg
```

## Region selection and GPU availability

Region selection limits the pool of instances we can run your Functions on. As a result, you may observe higher wait times between when your Function is called and when it gets executed. Generally, we have higher availability in US/EU versus other regions. Whenever possible, select the broadest possible regions so you get the best resource availability.

#### Container lifecycle hooks

# Container lifecycle hooks

Since Modal will reuse the same container for multiple inputs, sometimes you
might want to run some code exactly once when the container starts or exits.

To accomplish this, you need to use Modal's class syntax and the
[`@app.cls`](https://modal.com/docs/reference/modal.App#cls) decorator. Specifically, you'll
need to:

1. Convert your function to a method by making it a member of a class.
2. Decorate the class with `@app.cls(...)` with same arguments you previously
   had for `@app.function(...)`.
3. Instead of the `@app.function` decorator on the original method, use
   `@method` or the appropriate decorator for a
   [web endpoint](#lifecycle-hooks-for-web-endpoints).
4. Add the correct method "hooks" to your class based on your need:
   - `@enter` for one-time initialization (remote)
   - `@exit` for one-time cleanup (remote)

## `@enter`

The container entry handler is called when a new container is started. This is
useful for doing one-time initialization, such as loading model weights or
importing packages that are only present in that image.

To use, make your function a member of a class, and apply the `@enter()`
decorator to one or more class methods:

```python
import modal

app = modal.App()

@app.cls(cpu=8)
class Model:
    @modal.enter()
    def run_this_on_container_startup(self):
        import pickle
        self.model = pickle.load(open("model.pickle"))

    @modal.method()
    def predict(self, x):
        return self.model.predict(x)

@app.local_entrypoint()
def main():
    Model().predict.remote(x=123)
```

When working with an [asynchronous Modal](https://modal.com/docs/guide/async) app, you may use an
async method instead:

```python
import modal

app = modal.App()

@app.cls(memory=1024)
class Processor:
    @modal.enter()
    async def my_enter_method(self):
        self.cache = await load_cache()

    @modal.method()
    async def run(self, x):
        return await do_some_async_stuff(x, self.cache)

@app.local_entrypoint()
async def main():
    await Processor().run.remote(x=123)
```

Note: The `@enter()` decorator replaces the earlier `__enter__` syntax, which
has been deprecated.

## `@exit`

The container exit handler is called when a container is about to exit. It is
useful for doing one-time cleanup, such as closing a database connection or
saving intermediate results. To use, make your function a member of a class, and
apply the `@exit()` decorator:

```python
import modal

app = modal.App()

@app.cls()
class ETLPipeline:
    @modal.enter()
    def open_connection(self):
        import psycopg2
        self.connection = psycopg2.connect(os.environ["DATABASE_URI"])

    @modal.method()
    def run(self):
        # Run some queries
        pass

    @modal.exit()
    def close_connection(self):
        self.connection.close()

@app.local_entrypoint()
def main():
    ETLPipeline().run.remote()
```

Exit handlers are also called when a container is [preempted](https://modal.com/docs/guide/preemption).
The exit handler is given a grace period of 30 seconds to finish, and it will be
killed if it takes longer than that to complete.

## Lifecycle hooks for web endpoints

Modal `@function`s that are [web endpoints](https://modal.com/docs/guide/webhooks) can be
converted to the class syntax as well. Instead of `@modal.method`, simply use
whichever of the web endpoint decorators (`@modal.fastapi_endpoint`,
`@modal.asgi_app` or `@modal.wsgi_app`) you were using before.

```python
from fastapi import Request

import modal

image = modal.Image.debian_slim().pip_install("fastapi")
app = modal.App("web-endpoint-cls", image=image)

@app.cls()
class Model:
    @modal.enter()
    def run_this_on_container_startup(self):
        self.model = pickle.load(open("model.pickle"))

    @modal.fastapi_endpoint()
    def predict(self, request: Request):
        ...
```

#### Parametrized functions

# Parametrized functions

A single Modal Function can be parametrized by a set of arguments, so that each unique combination of arguments will behave like an individual
Modal Function with its own auto-scaling and lifecycle logic.

For example, you might want to have a separate pool of containers for each unique user that invokes your Function. In this scenario, you would
parametrize your Function by a user ID.

To parametrize a Modal Function, you need to use Modal's [class syntax](https://modal.com/docs/guide/lifecycle-functions) and the
[`@app.cls`](https://modal.com/docs/reference/modal.App#cls) decorator. Specifically, you'll need to:

1. Convert your function to a method by making it a member of a class.
2. Decorate the class with `@app.cls(...)` with the same arguments you previously
   had for `@app.function(...)` or your [web endpoint decorator](https://modal.com/docs/guide/webhooks).
3. If you previously used the `@app.function()` decorator on your function, replace it with `@modal.method()`.
4. Define dataclass-style, type-annotated instance attributes with `modal.parameter()` and optionally set default values:

```python
import modal

app = modal.App()

@app.cls()
class MyClass:

    foo: str = modal.parameter()
    bar: int = modal.parameter(default=10)

    @modal.method()
    def baz(self, qux: str = "default") -> str:
        return f"This code is running in container pool ({self.foo}, {self.bar}), with input qux={qux}"
```

The parameters create a keyword-only constructor for your class, and the methods can be called as follows:

```python
@app.local_entrypoint()
def main():
    m1 = MyClass(foo="hedgehog", bar=7)
    m1.baz.remote()

    m2 = MyClass(foo="fox")
    m2.baz.remote(qux="override")
```

Function calls for each unique combination of values for `foo` and `bar` will run in their own separate container pools.
If you re-constructed a `MyClass` with the same arguments in a different context, the calls to `baz` would be routed to the same set of containers as before.

Some things to note:

- The total size of the arguments is limited to 16 KiB.
- Modal classes can still annotate types of regular class attributes, which are independent of parametrization, by either omitting `= modal.parameter()` or using `= modal.parameter(init=False)` to satisfy type checkers.
- The support types are these primitives: `str`, `int`, `bool`, and `bytes`.
- The legacy `__init__` constructor method is being removed, see [the 1.0 migration for details.](https://modal.com/docs/guide/modal-1-0-migration#removing-support-for-custom-cls-constructors)

## Looking up a parametrized function

If you want to call your parametrized function from a Python script running
anywhere, you can use `Cls.lookup`:

```python notest
import modal

MyClass = modal.Cls.from_name("parametrized-function-app", "MyClass")  # returns a class-like object
m = MyClass(foo="snake", bar=12)
m.baz.remote()
```

## Parametrized web endpoints

Modal [web endpoints](https://modal.com/docs/guide/webhooks) can also be parametrized:

```python
app = modal.App("parametrized-endpoint")

@app.cls()
class MyClass():

    foo: str = modal.parameter()
    bar: int = modal.parameter(default=10)

    @modal.fastapi_endpoint()
    def baz(self, qux: str = "default") -> str:
        ...
```

Parameters are specified in the URL as query parameter values.

```bash
curl "https://parametrized-endpoint.modal.run?foo=hedgehog&bar=7&qux=override"
curl "https://parametrized-endpoint.modal.run?foo=hedgehog&qux=override"
curl "https://parametrized-endpoint.modal.run?foo=hedgehog&bar=7"
curl "https://parametrized-endpoint.modal.run?foo=hedgehog"
```

## Using parametrized functions with lifecycle functions

Parametrized functions can be used with [lifecycle functions](https://modal.com/docs/guide/lifecycle-functions).

For example, here is how you might parametrize the [`@enter`](https://modal.com/docs/guide/lifecycle-functions#enter) lifecycle function to load a specific model:

```python
@app.cls()
class Model:

    name: str = modal.parameter()
    size: int = modal.parameter(default=100)

    @modal.enter()
    def load_model(self):
        print(f"Loading model {self.name} with size {self.size}")
        self.model = load_model_util(self.name, self.size)

    @modal.method()
    def generate(self, prompt: str) -> str:
        return self.model.generate(prompt)
```

## Performance

Currently, parametrized Function creation is rate limited to 1 per second, with the ability to burst to 1000. Please [get in touch](mailto:support@modal.com) if you need higher rate limits.

#### S3 Gateway endpoints

# S3 Gateway endpoints

When running workloads in AWS, our system automatically uses a corresponding
[S3 Gateway endpoint](https://docs.aws.amazon.com/vpc/latest/privatelink/vpc-endpoints-s3.html)
to ensure low costs, optimal performance, and network reliability between Modal and S3.

Workloads running on Modal should not incur egress or ingress fees associated
with S3 operations. No configuration is needed in order for your app to use S3 Gateway endpoints.
S3 Gateway endpoints are automatically used when your app runs on AWS.

## Endpoint configuration

Only use the region-specific endpoint (`s3.<region>.amazonaws.com`) or the
global AWS endpoint (`s3.amazonaws.com`). Using an S3 endpoint from one region
in another **will not use the S3 Gateway Endpoint incurring networking costs**.

Avoid specifying regional endpoints manually, as this can lead to unexpected cost
or performance degradation.

## Inter-region costs

S3 Gateway endpoints guarantee no costs for network traffic within the same AWS region.
However, if your Modal Function runs in one region but your bucket resides in a
different region you will be billed for inter-region traffic.

You can prevent this by scheduling your Modal App in the same region of your
S3 bucket with [Region selection](https://modal.com/docs/guide/region-selection#region-selection).

#### GPU Metrics

# GPU Metrics

Modal exposes a number of GPU metrics that help monitor the health and utilization of the GPUs you're using.

- **GPU utilization %** is the percentage of time that the GPU was executing at least one CUDA kernel. This is the same metric reported as utilization by [`nvidia-smi`](https://modal.com/gpu-glossary/host-software/nvidia-smi). GPU utilization is helpful for determining the amount of time GPU work is blocked on CPU work, like PyTorch compute graph construction or input processing. However, it is far from indicating what fraction of the GPU's computing firepower (FLOPS or memory throughput, [CUDA Cores](https://modal.com/gpu-glossary/device-hardware/cuda-core), [SMs](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor)) is being used. See [this blog post](https://arthurchiao.art/blog/understanding-gpu-performance) for details.
- **GPU power utilization %** is the percentage of the maximum power draw that the device is currently drawing. When aggregating across containers, we also report **Total GPU power usage** in Watts. Because high-performance GPUs are [fundamentally limited by power draw](https://www.thonking.ai/p/strangely-matrix-multiplications), both for computation and memory access, the power usage can be used as a proxy of how much work the GPU is doing. A fully-saturated GPU should draw at or near its entire power budget (which can also be found by running `nvidia-smi`).
- **GPU temperature** is the temperature measured on the die of the GPU. Like power draw, which is the source of the thermal energy, the ability to efflux heat is a fundamental limit on GPU performance: continuing to draw full power without removing the waste heat would damage the system. At the highest temperatures readily observed in proper GPU deployments (i.e. mid-70s Celsius for an H100), increased error correction from thermal noise can already reduce performance. Generally, power utilization is a better proxy for performance, but we report temperature for completeness.
- **GPU memory used** is the amount of memory allocated on the GPU, in bytes.

In general, these metrics are useful signals or correlates of performance, but can't be used to directly debug performance issues. Instead, we (and [the manufacturers!](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#assess-parallelize-optimize-deploy)) recommend tracing and profiling workloads. See [this example](https://modal.com/docs/examples/torch_profiling) of profiling PyTorch applications on Modal.

## API Reference

### Python API Reference

#### App

# modal.App

```python
class App(object)
```

A Modal App is a group of functions and classes that are deployed together.

The app serves at least three purposes:

* A unit of deployment for functions and classes.
* Syncing of identities of (primarily) functions and classes across processes
  (your local Python interpreter and every Modal container active in your application).
* Manage log collection for everything that happens inside your code.

**Registering functions with an app**

The most common way to explicitly register an Object with an app is through the
`@app.function()` decorator. It both registers the annotated function itself and
other passed objects, like schedules and secrets, with the app:

```python
import modal

app = modal.App()

@app.function(
    secrets=[modal.Secret.from_name("some_secret")],
    schedule=modal.Period(days=1),
)
def foo():
    pass
```

In this example, the secret and schedule are registered with the app.

```python
def __init__(
    self,
    name: Optional[str] = None,
    *,
    image: Optional[_Image] = None,  # Default Image for the App (otherwise default to `modal.Image.debian_slim()`)
    secrets: Sequence[_Secret] = [],  # Secrets to add for all Functions in the App
    volumes: dict[Union[str, PurePosixPath], _Volume] = {},  # Volume mounts to use for all Functions
    include_source: bool = True,  # Default configuration for adding Function source file(s) to the Modal container
) -> None:
```

Construct a new app, optionally with default image, mounts, secrets, or volumes.

```python notest
image = modal.Image.debian_slim().pip_install(...)
secret = modal.Secret.from_name("my-secret")
volume = modal.Volume.from_name("my-data")
app = modal.App(image=image, secrets=[secret], volumes={"/mnt/data": volume})
```
## name

```python
@property
def name(self) -> Optional[str]:
```

The user-provided name of the App.
## is_interactive

```python
@property
def is_interactive(self) -> bool:
```

Whether the current app for the app is running in interactive mode.
## app_id

```python
@property
def app_id(self) -> Optional[str]:
```

Return the app_id of a running or stopped app.
## description

```python
@property
def description(self) -> Optional[str]:
```

The App's `name`, if available, or a fallback descriptive identifier.
## lookup

```python
@staticmethod
def lookup(
    name: str,
    *,
    client: Optional[_Client] = None,
    environment_name: Optional[str] = None,
    create_if_missing: bool = False,
) -> "_App":
```

Look up an App with a given name, creating a new App if necessary.

Note that Apps created through this method will be in a deployed state,
but they will not have any associated Functions or Classes. This method
is mainly useful for creating an App to associate with a Sandbox:

```python
app = modal.App.lookup("my-app", create_if_missing=True)
modal.Sandbox.create("echo", "hi", app=app)
```
## set_description

```python
def set_description(self, description: str):
```

## image

```python
@property
def image(self) -> _Image:
```

## run

```python
@contextmanager
def run(
    self,
    *,
    client: Optional[_Client] = None,
    detach: bool = False,
    interactive: bool = False,
    environment_name: Optional[str] = None,
) -> AsyncGenerator["_App", None]:
```

Context manager that runs an ephemeral app on Modal.

Use this as the main entry point for your Modal application. All calls
to Modal Functions should be made within the scope of this context
manager, and they will correspond to the current App.

**Example**

```python notest
with app.run():
    some_modal_function.remote()
```

To enable output printing (i.e., to see App logs), use `modal.enable_output()`:

```python notest
with modal.enable_output():
    with app.run():
        some_modal_function.remote()
```

Note that you should not invoke this in global scope of a file where you have
Modal Functions or Classes defined, since that would run the block when the Function
or Cls is imported in your containers as well. If you want to run it as your entrypoint,
consider protecting it:

```python
if __name__ == "__main__":
    with app.run():
        some_modal_function.remote()
```

You can then run your script with:

```shell
python app_module.py
```
## deploy

```python
def deploy(
    self,
    *,
    name: Optional[str] = None,  # Name for the deployment, overriding any set on the App
    environment_name: Optional[str] = None,  # Environment to deploy the App in
    tag: str = "",  # Optional metadata that will be visible in the deployment history
    client: Optional[_Client] = None,  # Alternate client to use for RPCs
) -> typing_extensions.Self:
```

Deploy the App so that it is available persistently.

Deployed Apps will be avaible for lookup or web-based invocations until they are stopped.
Unlike with `App.run`, this method will return as soon as the deployment completes.

This method is a programmatic alternative to the `modal deploy` CLI command.

Examples:

```python notest
app = App("my-app")
app.deploy()
```

To enable output printing (i.e., to see build logs), use `modal.enable_output()`:

```python notest
app = App("my-app")
with modal.enable_output():
    app.deploy()
```

Unlike with `App.run`, Function logs will not stream back to the local client after the
App is deployed.

Note that you should not invoke this method in global scope, as that would redeploy
the App every time the file is imported. If you want to write a programmatic deployment
script, protect this call so that it only runs when the file is executed directly:

```python notest
if __name__ == "__main__":
    with modal.enable_output():
        app.deploy()
```

Then you can deploy your app with:

```shell
python app_module.py
```
## registered_functions

```python
@property
def registered_functions(self) -> dict[str, _Function]:
```

All modal.Function objects registered on the app.

Note: this property is populated only during the build phase, and it is not
expected to work when a deplyoed App has been retrieved via `modal.App.lookup`.
## registered_classes

```python
@property
def registered_classes(self) -> dict[str, _Cls]:
```

All modal.Cls objects registered on the app.

Note: this property is populated only during the build phase, and it is not
expected to work when a deplyoed App has been retrieved via `modal.App.lookup`.
## registered_entrypoints

```python
@property
def registered_entrypoints(self) -> dict[str, _LocalEntrypoint]:
```

All local CLI entrypoints registered on the app.

Note: this property is populated only during the build phase, and it is not
expected to work when a deplyoed App has been retrieved via `modal.App.lookup`.
## registered_web_endpoints

```python
@property
def registered_web_endpoints(self) -> list[str]:
```

Names of web endpoint (ie. webhook) functions registered on the app.

Note: this property is populated only during the build phase, and it is not
expected to work when a deplyoed App has been retrieved via `modal.App.lookup`.
## local_entrypoint

```python
def local_entrypoint(
    self, _warn_parentheses_missing: Any = None, *, name: Optional[str] = None
) -> Callable[[Callable[..., Any]], _LocalEntrypoint]:
```

Decorate a function to be used as a CLI entrypoint for a Modal App.

These functions can be used to define code that runs locally to set up the app,
and act as an entrypoint to start Modal functions from. Note that regular
Modal functions can also be used as CLI entrypoints, but unlike `local_entrypoint`,
those functions are executed remotely directly.

**Example**

```python
@app.local_entrypoint()
def main():
    some_modal_function.remote()
```

You can call the function using `modal run` directly from the CLI:

```shell
modal run app_module.py
```

Note that an explicit [`app.run()`](https://modal.com/docs/reference/modal.App#run) is not needed, as an
[app](https://modal.com/docs/guide/apps) is automatically created for you.

**Multiple Entrypoints**

If you have multiple `local_entrypoint` functions, you can qualify the name of your app and function:

```shell
modal run app_module.py::app.some_other_function
```

**Parsing Arguments**

If your entrypoint function take arguments with primitive types, `modal run` automatically parses them as
CLI options.
For example, the following function can be called with `modal run app_module.py --foo 1 --bar "hello"`:

```python
@app.local_entrypoint()
def main(foo: int, bar: str):
    some_modal_function.call(foo, bar)
```

Currently, `str`, `int`, `float`, `bool`, and `datetime.datetime` are supported.
Use `modal run app_module.py --help` for more information on usage.
## function

```python
@warn_on_renamed_autoscaler_settings
def function(
    self,
    *,
    image: Optional[_Image] = None,  # The image to run as the container for the function
    schedule: Optional[Schedule] = None,  # An optional Modal Schedule for the function
    secrets: Sequence[_Secret] = (),  # Optional Modal Secret objects with environment variables for the container
    gpu: Union[
        GPU_T, list[GPU_T]
    ] = None,  # GPU request as string ("any", "T4", ...), object (`modal.GPU.A100()`, ...), or a list of either
    serialized: bool = False,  # Whether to send the function over using cloudpickle.
    network_file_systems: dict[
        Union[str, PurePosixPath], _NetworkFileSystem
    ] = {},  # Mountpoints for Modal NetworkFileSystems
    volumes: dict[
        Union[str, PurePosixPath], Union[_Volume, _CloudBucketMount]
    ] = {},  # Mount points for Modal Volumes & CloudBucketMounts
    # Specify, in fractional CPU cores, how many CPU cores to request.
    # Or, pass (request, limit) to additionally specify a hard limit in fractional CPU cores.
    # CPU throttling will prevent a container from exceeding its specified limit.
    cpu: Optional[Union[float, tuple[float, float]]] = None,
    # Specify, in MiB, a memory request which is the minimum memory required.
    # Or, pass (request, limit) to additionally specify a hard limit in MiB.
    memory: Optional[Union[int, tuple[int, int]]] = None,
    ephemeral_disk: Optional[int] = None,  # Specify, in MiB, the ephemeral disk size for the Function.
    min_containers: Optional[int] = None,  # Minimum number of containers to keep warm, even when Function is idle.
    max_containers: Optional[int] = None,  # Limit on the number of containers that can be concurrently running.
    buffer_containers: Optional[int] = None,  # Number of additional idle containers to maintain under active load.
    scaledown_window: Optional[int] = None,  # Max time (in seconds) a container can remain idle while scaling down.
    proxy: Optional[_Proxy] = None,  # Reference to a Modal Proxy to use in front of this function.
    retries: Optional[Union[int, Retries]] = None,  # Number of times to retry each input in case of failure.
    timeout: int = 300,  # Maximum execution time for inputs and startup time in seconds.
    startup_timeout: Optional[int] = None,  # Maximum startup time in seconds with higher precedence than `timeout`.
    name: Optional[str] = None,  # Sets the Modal name of the function within the app
    is_generator: Optional[
        bool
    ] = None,  # Set this to True if it's a non-generator function returning a [sync/async] generator object
    cloud: Optional[str] = None,  # Cloud provider to run the function on. Possible values are aws, gcp, oci, auto.
    region: Optional[Union[str, Sequence[str]]] = None,  # Region or regions to run the function on.
    enable_memory_snapshot: bool = False,  # Enable memory checkpointing for faster cold starts.
    block_network: bool = False,  # Whether to block network access
    restrict_modal_access: bool = False,  # Whether to allow this function access to other Modal resources
    # Maximum number of inputs a container should handle before shutting down.
    # With `max_inputs = 1`, containers will be single-use.
    max_inputs: Optional[int] = None,
    i6pn: Optional[bool] = None,  # Whether to enable IPv6 container networking within the region.
    # Whether the file or directory containing the Function's source should automatically be included
    # in the container. When unset, falls back to the App-level configuration, or is otherwise True by default.
    include_source: Optional[bool] = None,
    experimental_options: Optional[dict[str, Any]] = None,
    # Parameters below here are experimental. Use with caution!
    _experimental_scheduler_placement: Optional[
        SchedulerPlacement
    ] = None,  # Experimental controls over fine-grained scheduling (alpha).
    _experimental_proxy_ip: Optional[str] = None,  # IP address of proxy
    _experimental_custom_scaling_factor: Optional[float] = None,  # Custom scaling factor
    # Parameters below here are deprecated. Please update your code as suggested
    keep_warm: Optional[int] = None,  # Replaced with `min_containers`
    concurrency_limit: Optional[int] = None,  # Replaced with `max_containers`
    container_idle_timeout: Optional[int] = None,  # Replaced with `scaledown_window`
    allow_concurrent_inputs: Optional[int] = None,  # Replaced with the `@modal.concurrent` decorator
    allow_cross_region_volumes: Optional[bool] = None,  # Always True on the Modal backend now
    _experimental_buffer_containers: Optional[int] = None,  # Now stable API with `buffer_containers`
) -> _FunctionDecoratorType:
```

Decorator to register a new Modal Function with this App.
## cls

```python
@typing_extensions.dataclass_transform(field_specifiers=(parameter,), kw_only_default=True)
@warn_on_renamed_autoscaler_settings
def cls(
    self,
    *,
    image: Optional[_Image] = None,  # The image to run as the container for the function
    secrets: Sequence[_Secret] = (),  # Optional Modal Secret objects with environment variables for the container
    gpu: Union[
        GPU_T, list[GPU_T]
    ] = None,  # GPU request as string ("any", "T4", ...), object (`modal.GPU.A100()`, ...), or a list of either
    serialized: bool = False,  # Whether to send the function over using cloudpickle.
    network_file_systems: dict[
        Union[str, PurePosixPath], _NetworkFileSystem
    ] = {},  # Mountpoints for Modal NetworkFileSystems
    volumes: dict[
        Union[str, PurePosixPath], Union[_Volume, _CloudBucketMount]
    ] = {},  # Mount points for Modal Volumes & CloudBucketMounts
    # Specify, in fractional CPU cores, how many CPU cores to request.
    # Or, pass (request, limit) to additionally specify a hard limit in fractional CPU cores.
    # CPU throttling will prevent a container from exceeding its specified limit.
    cpu: Optional[Union[float, tuple[float, float]]] = None,
    # Specify, in MiB, a memory request which is the minimum memory required.
    # Or, pass (request, limit) to additionally specify a hard limit in MiB.
    memory: Optional[Union[int, tuple[int, int]]] = None,
    ephemeral_disk: Optional[int] = None,  # Specify, in MiB, the ephemeral disk size for the Function.
    min_containers: Optional[int] = None,  # Minimum number of containers to keep warm, even when Function is idle.
    max_containers: Optional[int] = None,  # Limit on the number of containers that can be concurrently running.
    buffer_containers: Optional[int] = None,  # Number of additional idle containers to maintain under active load.
    scaledown_window: Optional[int] = None,  # Max time (in seconds) a container can remain idle while scaling down.
    proxy: Optional[_Proxy] = None,  # Reference to a Modal Proxy to use in front of this function.
    retries: Optional[Union[int, Retries]] = None,  # Number of times to retry each input in case of failure.
    timeout: int = 300,  # Maximum execution time for inputs and startup time in seconds.
    startup_timeout: Optional[int] = None,  # Maximum startup time in seconds with higher precedence than `timeout`.
    cloud: Optional[str] = None,  # Cloud provider to run the function on. Possible values are aws, gcp, oci, auto.
    region: Optional[Union[str, Sequence[str]]] = None,  # Region or regions to run the function on.
    enable_memory_snapshot: bool = False,  # Enable memory checkpointing for faster cold starts.
    block_network: bool = False,  # Whether to block network access
    restrict_modal_access: bool = False,  # Whether to allow this class access to other Modal resources
    # Limits the number of inputs a container handles before shutting down.
    # Use `max_inputs = 1` for single-use containers.
    max_inputs: Optional[int] = None,
    i6pn: Optional[bool] = None,  # Whether to enable IPv6 container networking within the region.
    include_source: Optional[bool] = None,  # When `False`, don't automatically add the App source to the container.
    experimental_options: Optional[dict[str, Any]] = None,
    # Parameters below here are experimental. Use with caution!
    _experimental_scheduler_placement: Optional[
        SchedulerPlacement
    ] = None,  # Experimental controls over fine-grained scheduling (alpha).
    _experimental_proxy_ip: Optional[str] = None,  # IP address of proxy
    _experimental_custom_scaling_factor: Optional[float] = None,  # Custom scaling factor
    # Parameters below here are deprecated. Please update your code as suggested
    keep_warm: Optional[int] = None,  # Replaced with `min_containers`
    concurrency_limit: Optional[int] = None,  # Replaced with `max_containers`
    container_idle_timeout: Optional[int] = None,  # Replaced with `scaledown_window`
    allow_concurrent_inputs: Optional[int] = None,  # Replaced with the `@modal.concurrent` decorator
    _experimental_buffer_containers: Optional[int] = None,  # Now stable API with `buffer_containers`
    allow_cross_region_volumes: Optional[bool] = None,  # Always True on the Modal backend now
) -> Callable[[Union[CLS_T, _PartialFunction]], CLS_T]:
```

Decorator to register a new Modal [Cls](https://modal.com/docs/reference/modal.Cls) with this App.
## include

```python
def include(self, /, other_app: "_App") -> typing_extensions.Self:
```

Include another App's objects in this one.

Useful for splitting up Modal Apps across different self-contained files.

```python
app_a = modal.App("a")
@app.function()
def foo():
    ...

app_b = modal.App("b")
@app.function()
def bar():
    ...

app_a.include(app_b)

@app_a.local_entrypoint()
def main():
    # use function declared on the included app
    bar.remote()
```

#### Client

# modal.Client

```python
class Client(object)
```

## is_closed

```python
def is_closed(self) -> bool:
```

## hello

```python
def hello(self):
```

Connect to server and retrieve version information; raise appropriate error for various failures.
## from_credentials

```python
@classmethod
def from_credentials(cls, token_id: str, token_secret: str) -> "_Client":
```

Constructor based on token credentials; useful for managing Modal on behalf of third-party users.

**Usage:**

```python notest
client = modal.Client.from_credentials("my_token_id", "my_token_secret")

modal.Sandbox.create("echo", "hi", client=client, app=app)
```
## get_input_plane_metadata

```python
def get_input_plane_metadata(self, input_plane_region: str) -> list[tuple[str, str]]:
```

#### CloudBucketMount

# modal.CloudBucketMount

```python
class CloudBucketMount(object)
```

Mounts a cloud bucket to your container. Currently supports AWS S3 buckets.

S3 buckets are mounted using [AWS S3 Mountpoint](https://github.com/awslabs/mountpoint-s3).
S3 mounts are optimized for reading large files sequentially. It does not support every file operation; consult
[the AWS S3 Mountpoint documentation](https://github.com/awslabs/mountpoint-s3/blob/main/doc/SEMANTICS.md)
for more information.

**AWS S3 Usage**

```python
import subprocess

app = modal.App()
secret = modal.Secret.from_name(
    "aws-secret",
    required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    # Note: providing AWS_REGION can help when automatic detection of the bucket region fails.
)

@app.function(
    volumes={
        "/my-mount": modal.CloudBucketMount(
            bucket_name="s3-bucket-name",
            secret=secret,
            read_only=True
        )
    }
)
def f():
    subprocess.run(["ls", "/my-mount"], check=True)
```

**Cloudflare R2 Usage**

Cloudflare R2 is [S3-compatible](https://developers.cloudflare.com/r2/api/s3/api/) so its setup looks
very similar to S3. But additionally the `bucket_endpoint_url` argument must be passed.

```python
import subprocess

app = modal.App()
secret = modal.Secret.from_name(
    "r2-secret",
    required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
)

@app.function(
    volumes={
        "/my-mount": modal.CloudBucketMount(
            bucket_name="my-r2-bucket",
            bucket_endpoint_url="https://<ACCOUNT ID>.r2.cloudflarestorage.com",
            secret=secret,
            read_only=True
        )
    }
)
def f():
    subprocess.run(["ls", "/my-mount"], check=True)
```

**Google GCS Usage**

Google Cloud Storage (GCS) is [S3-compatible](https://cloud.google.com/storage/docs/interoperability).
GCS Buckets also require a secret with Google-specific key names (see below) populated with
a [HMAC key](https://cloud.google.com/storage/docs/authentication/managing-hmackeys#create).

```python
import subprocess

app = modal.App()
gcp_hmac_secret = modal.Secret.from_name(
    "gcp-secret",
    required_keys=["GOOGLE_ACCESS_KEY_ID", "GOOGLE_ACCESS_KEY_SECRET"]
)

@app.function(
    volumes={
        "/my-mount": modal.CloudBucketMount(
            bucket_name="my-gcs-bucket",
            bucket_endpoint_url="https://storage.googleapis.com",
            secret=gcp_hmac_secret,
        )
    }
)
def f():
    subprocess.run(["ls", "/my-mount"], check=True)
```

```python
def __init__(self, bucket_name: str, bucket_endpoint_url: Optional[str] = None, key_prefix: Optional[str] = None, secret: Optional[modal.secret._Secret] = None, oidc_auth_role_arn: Optional[str] = None, read_only: bool = False, requester_pays: bool = False) -> None
```

#### Cls

# modal.Cls

```python
class Cls(modal.object.Object)
```

Cls adds method pooling and [lifecycle hook](https://modal.com/docs/guide/lifecycle-functions) behavior
to [modal.Function](https://modal.com/docs/reference/modal.Function).

Generally, you will not construct a Cls directly.
Instead, use the [`@app.cls()`](https://modal.com/docs/reference/modal.App#cls) decorator on the App object.

## hydrate

```python
def hydrate(self, client: Optional[_Client] = None) -> Self:
```

Synchronize the local object with its identity on the Modal server.

It is rarely necessary to call this method explicitly, as most operations
will lazily hydrate when needed. The main use case is when you need to
access object metadata, such as its ID.

*Added in v0.72.39*: This method replaces the deprecated `.resolve()` method.
## from_name

```python
@classmethod
def from_name(
    cls: type["_Cls"],
    app_name: str,
    name: str,
    *,
    environment_name: Optional[str] = None,
) -> "_Cls":
```

Reference a Cls from a deployed App by its name.

This is a lazy method that defers hydrating the local
object with metadata from Modal servers until the first
time it is actually used.

```python
Model = modal.Cls.from_name("other-app", "Model")
```
## with_options

```python
@warn_on_renamed_autoscaler_settings
def with_options(
    self: "_Cls",
    *,
    cpu: Optional[Union[float, tuple[float, float]]] = None,
    memory: Optional[Union[int, tuple[int, int]]] = None,
    gpu: GPU_T = None,
    secrets: Collection[_Secret] = (),
    volumes: dict[Union[str, os.PathLike], _Volume] = {},
    retries: Optional[Union[int, Retries]] = None,
    max_containers: Optional[int] = None,  # Limit on the number of containers that can be concurrently running.
    buffer_containers: Optional[int] = None,  # Additional containers to scale up while Function is active.
    scaledown_window: Optional[int] = None,  # Max amount of time a container can remain idle before scaling down.
    timeout: Optional[int] = None,
    region: Optional[Union[str, Sequence[str]]] = None,  # Region or regions to run the function on.
    cloud: Optional[str] = None,  # Cloud provider to run the function on. Possible values are aws, gcp, oci, auto.
    # The following parameters are deprecated
    concurrency_limit: Optional[int] = None,  # Now called `max_containers`
    container_idle_timeout: Optional[int] = None,  # Now called `scaledown_window`
    allow_concurrent_inputs: Optional[int] = None,  # See `.with_concurrency`
) -> "_Cls":
```

Override the static Function configuration at runtime.

This method will return a new instance of the cls that will autoscale independently of the
original instance. Note that options cannot be "unset" with this method (i.e., if a GPU
is configured in the `@app.cls()` decorator, passing `gpu=None` here will not create a
CPU-only instance).

**Usage:**

You can use this method after looking up the Cls from a deployed App or if you have a
direct reference to a Cls from another Function or local entrypoint on its App:

```python notest
Model = modal.Cls.from_name("my_app", "Model")
ModelUsingGPU = Model.with_options(gpu="A100")
ModelUsingGPU().generate.remote(input_prompt)  # Run with an A100 GPU
```

The method can be called multiple times to "stack" updates:

```python notest
Model.with_options(gpu="A100").with_options(scaledown_window=300)  # Use an A100 with slow scaledown
```

Note that container arguments (i.e. `volumes` and `secrets`) passed in subsequent calls
will not be merged.
## with_concurrency

```python
def with_concurrency(self: "_Cls", *, max_inputs: int, target_inputs: Optional[int] = None) -> "_Cls":
```

Create an instance of the Cls with input concurrency enabled or overridden with new values.

**Usage:**

```python notest
Model = modal.Cls.from_name("my_app", "Model")
ModelUsingGPU = Model.with_options(gpu="A100").with_concurrency(max_inputs=100)
ModelUsingGPU().generate.remote(42)  # will run on an A100 GPU with input concurrency enabled
```
## with_batching

```python
def with_batching(self: "_Cls", *, max_batch_size: int, wait_ms: int) -> "_Cls":
```

Create an instance of the Cls with dynamic batching enabled or overridden with new values.

**Usage:**

```python notest
Model = modal.Cls.from_name("my_app", "Model")
ModelUsingGPU = Model.with_options(gpu="A100").with_batching(max_batch_size=100, batch_wait_ms=1000)
ModelUsingGPU().generate.remote(42)  # will run on an A100 GPU with input concurrency enabled
```

#### Cron

# modal.Cron

```python
class Cron(modal.schedule.Schedule)
```

Cron jobs are a type of schedule, specified using the
[Unix cron tab](https://crontab.guru/) syntax.

The alternative schedule type is the [`modal.Period`](https://modal.com/docs/reference/modal.Period).

**Usage**

```python
import modal
app = modal.App()

@app.function(schedule=modal.Cron("* * * * *"))
def f():
    print("This function will run every minute")
```

We can specify different schedules with cron strings, for example:

```python
modal.Cron("5 4 * * *")  # run at 4:05am UTC every night
modal.Cron("0 9 * * 4")  # runs every Thursday at 9am UTC
```

We can also optionally specify a timezone, for example:

```python
# Run daily at 6am New York time, regardless of whether daylight saving
# is in effect (i.e. at 11am UTC in the winter, and 10am UTC in the summer):
modal.Cron("0 6 * * *", timezone="America/New_York")
```

If no timezone is specified, the default is UTC.

```python
def __init__(
    self,
    cron_string: str,
    timezone: str = "UTC",
) -> None:
```

Construct a schedule that runs according to a cron expression string.

#### Dict

# modal.Dict

```python
class Dict(modal.object.Object)
```

Distributed dictionary for storage in Modal apps.

Dict contents can be essentially any object so long as they can be serialized by
`cloudpickle`. This includes other Modal objects. If writing and reading in different
environments (eg., writing locally and reading remotely), it's necessary to have the
library defining the data type installed, with compatible versions, on both sides.
Additionally, cloudpickle serialization is not guaranteed to be deterministic, so it is
generally recommended to use primitive types for keys.

**Lifetime of a Dict and its items**

An individual Dict entry will expire after 7 days of inactivity (no reads or writes). The
Dict entries are written to durable storage.

Legacy Dicts (created before 2025-05-20) will still have entries expire 30 days after being
last added. Additionally, contents are stored in memory on the Modal server and could be lost
due to unexpected server restarts. Eventually, these Dicts will be fully sunset.

**Usage**

```python
from modal import Dict

my_dict = Dict.from_name("my-persisted_dict", create_if_missing=True)

my_dict["some key"] = "some value"
my_dict[123] = 456

assert my_dict["some key"] == "some value"
assert my_dict[123] == 456
```

The `Dict` class offers a few methods for operations that are usually accomplished
in Python with operators, such as `Dict.put` and `Dict.contains`. The advantage of
these methods is that they can be safely called in an asynchronous context by using
the `.aio` suffix on the method, whereas their operator-based analogues will always
run synchronously and block the event loop.

For more examples, see the [guide](https://modal.com/docs/guide/dicts-and-queues#modal-dicts).

## hydrate

```python
def hydrate(self, client: Optional[_Client] = None) -> Self:
```

Synchronize the local object with its identity on the Modal server.

It is rarely necessary to call this method explicitly, as most operations
will lazily hydrate when needed. The main use case is when you need to
access object metadata, such as its ID.

*Added in v0.72.39*: This method replaces the deprecated `.resolve()` method.
## objects

```python
class objects(object)
```

Namespace with methods for managing named Dict objects.

### create

```python
@staticmethod
def create(
    name: str,  # Name to use for the new Dict
    *,
    allow_existing: bool = False,  # If True, no-op when the Dict already exists
    environment_name: Optional[str] = None,  # Uses active environment if not specified
    client: Optional[_Client] = None,  # Optional client with Modal credentials
) -> None:
```

Create a new Dict object.

**Examples:**

```python notest
modal.Dict.objects.create("my-dict")
```

Dicts will be created in the active environment, or another one can be specified:

```python notest
modal.Dict.objects.create("my-dict", environment_name="dev")
```

By default, an error will be raised if the Dict already exists, but passing
`allow_existing=True` will make the creation attempt a no-op in this case.

```python notest
modal.Dict.objects.create("my-dict", allow_existing=True)
```

Note that this method does not return a local instance of the Dict. You can use
`modal.Dict.from_name` to perform a lookup after creation.

Added in v1.1.2.
### list

```python
@staticmethod
def list(
    *,
    max_objects: Optional[int] = None,  # Limit results to this size
    created_before: Optional[Union[datetime, str]] = None,  # Limit based on creation date
    environment_name: str = "",  # Uses active environment if not specified
    client: Optional[_Client] = None,  # Optional client with Modal credentials
) -> list["_Dict"]:
```

Return a list of hydrated Dict objects.

**Examples:**

```python
dicts = modal.Dict.objects.list()
print([d.name for d in dicts])
```

Dicts will be retreived from the active environment, or another one can be specified:

```python notest
dev_dicts = modal.Dict.objects.list(environment_name="dev")
```

By default, all named Dict are returned, newest to oldest. It's also possible to limit the
number of results and to filter by creation date:

```python
dicts = modal.Dict.objects.list(max_objects=10, created_before="2025-01-01")
```

Added in v1.1.2.
### delete

```python
@staticmethod
def delete(
    name: str,  # Name of the Dict to delete
    *,
    allow_missing: bool = False,  # If True, don't raise an error if the Dict doesn't exist
    environment_name: Optional[str] = None,  # Uses active environment if not specified
    client: Optional[_Client] = None,  # Optional client with Modal credentials
):
```

Delete a named Dict.

Warning: This deletes an *entire Dict*, not just a specific key.
Deletion is irreversible and will affect any Apps currently using the Dict.

**Examples:**

```python notest
await modal.Dict.objects.delete("my-dict")
```

Dicts will be deleted from the active environment, or another one can be specified:

```python notest
await modal.Dict.objects.delete("my-dict", environment_name="dev")
```

Added in v1.1.2.
## name

```python
@property
def name(self) -> Optional[str]:
```

## ephemeral

```python
@classmethod
@contextmanager
def ephemeral(
    cls: type["_Dict"],
    data: Optional[dict] = None,  # DEPRECATED
    client: Optional[_Client] = None,
    environment_name: Optional[str] = None,
) -> Iterator["_Dict"]:
```

Creates a new ephemeral Dict within a context manager:

Usage:
```python
from modal import Dict

with Dict.ephemeral() as d:
    d["foo"] = "bar"
```

```python notest
async with Dict.ephemeral() as d:
    await d.put.aio("foo", "bar")
```
## from_name

```python
@staticmethod
def from_name(
    name: str,
    *,
    environment_name: Optional[str] = None,
    create_if_missing: bool = False,
) -> "_Dict":
```

Reference a named Dict, creating if necessary.

This is a lazy method that defers hydrating the local
object with metadata from Modal servers until the first
time it is actually used.

```python
d = modal.Dict.from_name("my-dict", create_if_missing=True)
d[123] = 456
```
## info

```python
@live_method
def info(self) -> DictInfo:
```

Return information about the Dict object.
## clear

```python
@live_method
def clear(self) -> None:
```

Remove all items from the Dict.
## get

```python
@live_method
def get(self, key: Any, default: Optional[Any] = None) -> Any:
```

Get the value associated with a key.

Returns `default` if key does not exist.
## contains

```python
@live_method
def contains(self, key: Any) -> bool:
```

Return if a key is present.
## len

```python
@live_method
def len(self) -> int:
```

Return the length of the Dict.

Note: This is an expensive operation and will return at most 100,000.
## update

```python
@live_method
def update(self, other: Optional[Mapping] = None, /, **kwargs) -> None:
```

Update the Dict with additional items.
## put

```python
@live_method
def put(self, key: Any, value: Any, *, skip_if_exists: bool = False) -> bool:
```

Add a specific key-value pair to the Dict.

Returns True if the key-value pair was added and False if it wasn't because the key already existed and
`skip_if_exists` was set.
## pop

```python
@live_method
def pop(self, key: Any) -> Any:
```

Remove a key from the Dict, returning the value if it exists.
## keys

```python
@live_method_gen
def keys(self) -> Iterator[Any]:
```

Return an iterator over the keys in this Dict.

Note that (unlike with Python dicts) the return value is a simple iterator,
and results are unordered.
## values

```python
@live_method_gen
def values(self) -> Iterator[Any]:
```

Return an iterator over the values in this Dict.

Note that (unlike with Python dicts) the return value is a simple iterator,
and results are unordered.
## items

```python
@live_method_gen
def items(self) -> Iterator[tuple[Any, Any]]:
```

Return an iterator over the (key, value) tuples in this Dict.

Note that (unlike with Python dicts) the return value is a simple iterator,
and results are unordered.

#### Error

# modal.Error

```python
class Error(Exception)
```

Base class for all Modal errors. See [`modal.exception`](https://modal.com/docs/reference/modal.exception)
for the specialized error classes.

**Usage**

```python notest
import modal

try:
    ...
except modal.Error:
    # Catch any exception raised by Modal's systems.
    print("Responding to error...")
```

#### FilePatternMatcher

# modal.FilePatternMatcher

```python
class FilePatternMatcher(modal.file_pattern_matcher._AbstractPatternMatcher)
```

Allows matching file Path objects against a list of patterns.

**Usage:**
```python
from pathlib import Path
from modal import FilePatternMatcher

matcher = FilePatternMatcher("*.py")

assert matcher(Path("foo.py"))

# You can also negate the matcher.
negated_matcher = ~matcher

assert not negated_matcher(Path("foo.py"))
```

```python
def __init__(self, *pattern: str) -> None:
```

Initialize a new FilePatternMatcher instance.

Args:
    pattern (str): One or more pattern strings.

Raises:
    ValueError: If an illegal exclusion pattern is provided.
## can_prune_directories

```python
def can_prune_directories(self) -> bool:
```

Returns True if this pattern matcher allows safe early directory pruning.

Directory pruning is safe when matching directories can be skipped entirely
without missing any files that should be included. This is for example not
safe when we have inverted/negated ignore patterns (e.g. "!**/*.py").
## from_file

```python
@classmethod
def from_file(cls, file_path: Union[str, Path]) -> "FilePatternMatcher":
```

Initialize a new FilePatternMatcher instance from a file.

The patterns in the file will be read lazily when the matcher is first used.

Args:
    file_path (Path): The path to the file containing patterns.

**Usage:**
```python
from modal import FilePatternMatcher

matcher = FilePatternMatcher.from_file("/path/to/ignorefile")
```

#### Function

# modal.Function

```python
class Function(typing.Generic, modal.object.Object)
```

Functions are the basic units of serverless execution on Modal.

Generally, you will not construct a `Function` directly. Instead, use the
`App.function()` decorator to register your Python functions with your App.

## hydrate

```python
def hydrate(self, client: Optional[_Client] = None) -> Self:
```

Synchronize the local object with its identity on the Modal server.

It is rarely necessary to call this method explicitly, as most operations
will lazily hydrate when needed. The main use case is when you need to
access object metadata, such as its ID.

*Added in v0.72.39*: This method replaces the deprecated `.resolve()` method.
## update_autoscaler

```python
@live_method
def update_autoscaler(
    self,
    *,
    min_containers: Optional[int] = None,
    max_containers: Optional[int] = None,
    buffer_containers: Optional[int] = None,
    scaledown_window: Optional[int] = None,
) -> None:
```

Override the current autoscaler behavior for this Function.

Unspecified parameters will retain their current value, i.e. either the static value
from the function decorator, or an override value from a previous call to this method.

Subsequent deployments of the App containing this Function will reset the autoscaler back to
its static configuration.

Examples:

```python notest
f = modal.Function.from_name("my-app", "function")

# Always have at least 2 containers running, with an extra buffer when the Function is active
f.update_autoscaler(min_containers=2, buffer_containers=1)

# Limit this Function to avoid spinning up more than 5 containers
f.update_autoscaler(max_containers=5)

# Extend the scaledown window to increase the amount of time that idle containers stay alive
f.update_autoscaler(scaledown_window=300)

```
## from_name

```python
@classmethod
def from_name(
    cls: type["_Function"],
    app_name: str,
    name: str,
    *,
    environment_name: Optional[str] = None,
) -> "_Function":
```

Reference a Function from a deployed App by its name.

This is a lazy method that defers hydrating the local
object with metadata from Modal servers until the first
time it is actually used.

```python
f = modal.Function.from_name("other-app", "function")
```
## get_web_url

```python
@live_method
def get_web_url(self) -> Optional[str]:
```

URL of a Function running as a web endpoint.
## remote

```python
@live_method
def remote(self, *args: P.args, **kwargs: P.kwargs) -> ReturnType:
```

Calls the function remotely, executing it with the given arguments and returning the execution's result.
## remote_gen

```python
@live_method_gen
def remote_gen(self, *args, **kwargs) -> AsyncGenerator[Any, None]:
```

Calls the generator remotely, executing it with the given arguments and returning the execution's result.
## local

```python
def local(self, *args: P.args, **kwargs: P.kwargs) -> OriginalReturnType:
```

Calls the function locally, executing it with the given arguments and returning the execution's result.

The function will execute in the same environment as the caller, just like calling the underlying function
directly in Python. In particular, only secrets available in the caller environment will be available
through environment variables.
## spawn

```python
@live_method
def spawn(self, *args: P.args, **kwargs: P.kwargs) -> "_FunctionCall[ReturnType]":
```

Calls the function with the given arguments, without waiting for the results.

Returns a [`modal.FunctionCall`](https://modal.com/docs/reference/modal.FunctionCall) object
that can later be polled or waited for using
[`.get(timeout=...)`](https://modal.com/docs/reference/modal.FunctionCall#get).
Conceptually similar to `multiprocessing.pool.apply_async`, or a Future/Promise in other contexts.
## get_raw_f

```python
def get_raw_f(self) -> Callable[..., Any]:
```

Return the inner Python object wrapped by this Modal Function.
## get_current_stats

```python
@live_method
def get_current_stats(self) -> FunctionStats:
```

Return a `FunctionStats` object describing the current function's queue and runner counts.
## map

```python
@warn_if_generator_is_not_consumed(function_name="Function.map")
def map(
    self,
    *input_iterators: typing.Iterable[Any],  # one input iterator per argument in the mapped-over function/generator
    kwargs={},  # any extra keyword arguments for the function
    order_outputs: bool = True,  # return outputs in order
    return_exceptions: bool = False,  # propagate exceptions (False) or aggregate them in the results list (True)
    wrap_returned_exceptions: bool = True,
) -> AsyncOrSyncIterable:
```

Parallel map over a set of inputs.

Takes one iterator argument per argument in the function being mapped over.

Example:
```python
@app.function()
def my_func(a):
    return a ** 2

@app.local_entrypoint()
def main():
    assert list(my_func.map([1, 2, 3, 4])) == [1, 4, 9, 16]
```

If applied to a `app.function`, `map()` returns one result per input and the output order
is guaranteed to be the same as the input order. Set `order_outputs=False` to return results
in the order that they are completed instead.

`return_exceptions` can be used to treat exceptions as successful results:

```python
@app.function()
def my_func(a):
    if a == 2:
        raise Exception("ohno")
    return a ** 2

@app.local_entrypoint()
def main():
    # [0, 1, UserCodeException(Exception('ohno'))]
    print(list(my_func.map(range(3), return_exceptions=True)))
```
## starmap

```python
@warn_if_generator_is_not_consumed(function_name="Function.starmap")
def starmap(
    self,
    input_iterator: typing.Iterable[typing.Sequence[Any]],
    *,
    kwargs={},
    order_outputs: bool = True,
    return_exceptions: bool = False,
    wrap_returned_exceptions: bool = True,
) -> AsyncOrSyncIterable:
```

Like `map`, but spreads arguments over multiple function arguments.

Assumes every input is a sequence (e.g. a tuple).

Example:
```python
@app.function()
def my_func(a, b):
    return a + b

@app.local_entrypoint()
def main():
    assert list(my_func.starmap([(1, 2), (3, 4)])) == [3, 7]
```
## for_each

```python
def for_each(self, *input_iterators, kwargs={}, ignore_exceptions: bool = False):
```

Execute function for all inputs, ignoring outputs. Waits for completion of the inputs.

Convenient alias for `.map()` in cases where the function just needs to be called.
as the caller doesn't have to consume the generator to process the inputs.
## spawn_map

```python
def spawn_map(self, *input_iterators, kwargs={}) -> None:
```

Spawn parallel execution over a set of inputs, exiting as soon as the inputs are created (without waiting
for the map to complete).

Takes one iterator argument per argument in the function being mapped over.

Example:
```python
@app.function()
def my_func(a):
    return a ** 2

@app.local_entrypoint()
def main():
    my_func.spawn_map([1, 2, 3, 4])
```

Programmatic retrieval of results will be supported in a future update.

#### FunctionCall

# modal.FunctionCall

```python
class FunctionCall(typing.Generic, modal.object.Object)
```

A reference to an executed function call.

Constructed using `.spawn(...)` on a Modal function with the same
arguments that a function normally takes. Acts as a reference to
an ongoing function call that can be passed around and used to
poll or fetch function results at some later time.

Conceptually similar to a Future/Promise/AsyncResult in other contexts and languages.

## hydrate

```python
def hydrate(self, client: Optional[_Client] = None) -> Self:
```

Synchronize the local object with its identity on the Modal server.

It is rarely necessary to call this method explicitly, as most operations
will lazily hydrate when needed. The main use case is when you need to
access object metadata, such as its ID.

*Added in v0.72.39*: This method replaces the deprecated `.resolve()` method.
## num_inputs

```python
@live_method
def num_inputs(self) -> int:
```

Get the number of inputs in the function call.
## get

```python
def get(self, timeout: Optional[float] = None, *, index: int = 0) -> ReturnType:
```

Get the result of the index-th input of the function call.
`.spawn()` calls have a single output, so only specifying `index=0` is valid.
A non-zero index is useful when your function has multiple outputs, like via `.spawn_map()`.

This function waits indefinitely by default. It takes an optional
`timeout` argument that specifies the maximum number of seconds to wait,
which can be set to `0` to poll for an output immediately.

The returned coroutine is not cancellation-safe.
## iter

```python
@live_method_gen
def iter(self, *, start: int = 0, end: Optional[int] = None) -> Iterator[ReturnType]:
```

Iterate in-order over the results of the function call.

Optionally, specify a range [start, end) to iterate over.

Example:
```python
@app.function()
def my_func(a):
    return a ** 2

@app.local_entrypoint()
def main():
    fc = my_func.spawn_map([1, 2, 3, 4])
    assert list(fc.iter()) == [1, 4, 9, 16]
    assert list(fc.iter(start=1, end=3)) == [4, 9]
```

If `end` is not provided, it will iterate over all results.
## get_call_graph

```python
def get_call_graph(self) -> list[InputInfo]:
```

Returns a structure representing the call graph from a given root
call ID, along with the status of execution for each node.

See [`modal.call_graph`](https://modal.com/docs/reference/modal.call_graph) reference page
for documentation on the structure of the returned `InputInfo` items.
## cancel

```python
def cancel(
    self,
    # if true, containers running the inputs are forcibly terminated
    terminate_containers: bool = False,
):
```

Cancels the function call, which will stop its execution and mark its inputs as
[`TERMINATED`](https://modal.com/docs/reference/modal.call_graph#modalcall_graphinputstatus).

If `terminate_containers=True` - the containers running the cancelled inputs are all terminated
causing any non-cancelled inputs on those containers to be rescheduled in new containers.
## from_id

```python
@staticmethod
def from_id(function_call_id: str, client: Optional[_Client] = None) -> "_FunctionCall[Any]":
```

Instantiate a FunctionCall object from an existing ID.

Examples:

```python notest
# Spawn a FunctionCall and keep track of its object ID
fc = my_func.spawn()
fc_id = fc.object_id

# Later, use the ID to re-instantiate the FunctionCall object
fc = _FunctionCall.from_id(fc_id)
result = fc.get()
```

Note that it's only necessary to re-instantiate the `FunctionCall` with this method
if you no longer have access to the original object returned from `Function.spawn`.
## gather

```python
@staticmethod
def gather(*function_calls: "_FunctionCall[T]") -> typing.Sequence[T]:
```

Wait until all Modal FunctionCall objects have results before returning.

Accepts a variable number of `FunctionCall` objects, as returned by `Function.spawn()`.

Returns a list of results from each FunctionCall, or raises an exception
from the first failing function call.

Examples:

```python notest
fc1 = slow_func_1.spawn()
fc2 = slow_func_2.spawn()

result_1, result_2 = modal.FunctionCall.gather(fc1, fc2)
```

*Added in v0.73.69*: This method replaces the deprecated `modal.functions.gather` function.

#### Image

# modal.Image

```python
class Image(modal.object.Object)
```

Base class for container images to run functions in.

Do not construct this class directly; instead use one of its static factory methods,
such as `modal.Image.debian_slim`, `modal.Image.from_registry`, or `modal.Image.micromamba`.

## add_local_file

```python
def add_local_file(self, local_path: Union[str, Path], remote_path: str, *, copy: bool = False) -> "_Image":
```

Adds a local file to the image at `remote_path` within the container

By default (`copy=False`), the files are added to containers on startup and are not built into the actual Image,
which speeds up deployment.

Set `copy=True` to copy the files into an Image layer at build time instead, similar to how
[`COPY`](https://docs.docker.com/engine/reference/builder/#copy) works in a `Dockerfile`.

copy=True can slow down iteration since it requires a rebuild of the Image and any subsequent
build steps whenever the included files change, but it is required if you want to run additional
build steps after this one.

*Added in v0.66.40*: This method replaces the deprecated `modal.Image.copy_local_file` method.
## add_local_dir

```python
def add_local_dir(
    self,
    local_path: Union[str, Path],
    remote_path: str,
    *,
    copy: bool = False,
    # Predicate filter function for file exclusion, which should accept a filepath and return `True` for exclusion.
    # Defaults to excluding no files. If a Sequence is provided, it will be converted to a FilePatternMatcher.
    # Which follows dockerignore syntax.
    ignore: Union[Sequence[str], Callable[[Path], bool]] = [],
) -> "_Image":
```

Adds a local directory's content to the image at `remote_path` within the container

By default (`copy=False`), the files are added to containers on startup and are not built into the actual Image,
which speeds up deployment.

Set `copy=True` to copy the files into an Image layer at build time instead, similar to how
[`COPY`](https://docs.docker.com/engine/reference/builder/#copy) works in a `Dockerfile`.

copy=True can slow down iteration since it requires a rebuild of the Image and any subsequent
build steps whenever the included files change, but it is required if you want to run additional
build steps after this one.

**Usage:**

```python
from modal import FilePatternMatcher

image = modal.Image.debian_slim().add_local_dir(
    "~/assets",
    remote_path="/assets",
    ignore=["*.venv"],
)

image = modal.Image.debian_slim().add_local_dir(
    "~/assets",
    remote_path="/assets",
    ignore=lambda p: p.is_relative_to(".venv"),
)

image = modal.Image.debian_slim().add_local_dir(
    "~/assets",
    remote_path="/assets",
    ignore=FilePatternMatcher("**/*.txt"),
)

# When including files is simpler than excluding them, you can use the `~` operator to invert the matcher.
image = modal.Image.debian_slim().add_local_dir(
    "~/assets",
    remote_path="/assets",
    ignore=~FilePatternMatcher("**/*.py"),
)

# You can also read ignore patterns from a file.
image = modal.Image.debian_slim().add_local_dir(
    "~/assets",
    remote_path="/assets",
    ignore=FilePatternMatcher.from_file("/path/to/ignorefile"),
)
```

*Added in v0.66.40*: This method replaces the deprecated `modal.Image.copy_local_dir` method.
## add_local_python_source

```python
def add_local_python_source(
    self, *modules: str, copy: bool = False, ignore: Union[Sequence[str], Callable[[Path], bool]] = NON_PYTHON_FILES
) -> "_Image":
```

Adds locally available Python packages/modules to containers

Adds all files from the specified Python package or module to containers running the Image.

Packages are added to the `/root` directory of containers, which is on the `PYTHONPATH`
of any executed Modal Functions, enabling import of the module by that name.

By default (`copy=False`), the files are added to containers on startup and are not built into the actual Image,
which speeds up deployment.

Set `copy=True` to copy the files into an Image layer at build time instead. This can slow down iteration since
it requires a rebuild of the Image and any subsequent build steps whenever the included files change, but it is
required if you want to run additional build steps after this one.

**Note:** This excludes all dot-prefixed subdirectories or files and all `.pyc`/`__pycache__` files.
To add full directories with finer control, use `.add_local_dir()` instead and specify `/root` as
the destination directory.

By default only includes `.py`-files in the source modules. Set the `ignore` argument to a list of patterns
or a callable to override this behavior, e.g.:

```py
# includes everything except data.json
modal.Image.debian_slim().add_local_python_source("mymodule", ignore=["data.json"])

# exclude large files
modal.Image.debian_slim().add_local_python_source(
    "mymodule",
    ignore=lambda p: p.stat().st_size > 1e9
)
```

*Added in v0.67.28*: This method replaces the deprecated `modal.Mount.from_local_python_packages` pattern.
## from_id

```python
@staticmethod
def from_id(image_id: str, client: Optional[_Client] = None) -> "_Image":
```

Construct an Image from an id and look up the Image result.

The ID of an Image object can be accessed using `.object_id`.
## pip_install

```python
def pip_install(
    self,
    *packages: Union[str, list[str]],  # A list of Python packages, eg. ["numpy", "matplotlib>=3.5.0"]
    find_links: Optional[str] = None,  # Passes -f (--find-links) pip install
    index_url: Optional[str] = None,  # Passes -i (--index-url) to pip install
    extra_index_url: Optional[str] = None,  # Passes --extra-index-url to pip install
    pre: bool = False,  # Passes --pre (allow pre-releases) to pip install
    extra_options: str = "",  # Additional options to pass to pip install, e.g. "--no-build-isolation --no-clean"
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    secrets: Sequence[_Secret] = [],
    gpu: GPU_T = None,
) -> "_Image":
```

Install a list of Python packages using pip.

**Examples**

Simple installation:
```python
image = modal.Image.debian_slim().pip_install("click", "httpx~=0.23.3")
```

More complex installation:
```python
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
        "transformers==4.40.2",
    )
    .pip_install(
        "flash-attn==2.5.8", extra_options="--no-build-isolation"
    )
)
```
## pip_install_private_repos

```python
def pip_install_private_repos(
    self,
    *repositories: str,
    git_user: str,
    find_links: Optional[str] = None,  # Passes -f (--find-links) pip install
    index_url: Optional[str] = None,  # Passes -i (--index-url) to pip install
    extra_index_url: Optional[str] = None,  # Passes --extra-index-url to pip install
    pre: bool = False,  # Passes --pre (allow pre-releases) to pip install
    extra_options: str = "",  # Additional options to pass to pip install, e.g. "--no-build-isolation --no-clean"
    gpu: GPU_T = None,
    secrets: Sequence[_Secret] = [],
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
) -> "_Image":
```

Install a list of Python packages from private git repositories using pip.

This method currently supports Github and Gitlab only.

- **Github:** Provide a `modal.Secret` that contains a `GITHUB_TOKEN` key-value pair
- **Gitlab:** Provide a `modal.Secret` that contains a `GITLAB_TOKEN` key-value pair

These API tokens should have permissions to read the list of private repositories provided as arguments.

We recommend using Github's ['fine-grained' access tokens](https://github.blog/2022-10-18-introducing-fine-grained-personal-access-tokens-for-github/).
These tokens are repo-scoped, and avoid granting read permission across all of a user's private repos.

**Example**

```python
image = (
    modal.Image
    .debian_slim()
    .pip_install_private_repos(
        "github.com/ecorp/private-one@1.0.0",
        "github.com/ecorp/private-two@main"
        "github.com/ecorp/private-three@d4776502"
        # install from 'inner' directory on default branch.
        "github.com/ecorp/private-four#subdirectory=inner",
        git_user="erikbern",
        secrets=[modal.Secret.from_name("github-read-private")],
    )
)
```
## pip_install_from_requirements

```python
def pip_install_from_requirements(
    self,
    requirements_txt: str,  # Path to a requirements.txt file.
    find_links: Optional[str] = None,  # Passes -f (--find-links) pip install
    *,
    index_url: Optional[str] = None,  # Passes -i (--index-url) to pip install
    extra_index_url: Optional[str] = None,  # Passes --extra-index-url to pip install
    pre: bool = False,  # Passes --pre (allow pre-releases) to pip install
    extra_options: str = "",  # Additional options to pass to pip install, e.g. "--no-build-isolation --no-clean"
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    secrets: Sequence[_Secret] = [],
    gpu: GPU_T = None,
) -> "_Image":
```

Install a list of Python packages from a local `requirements.txt` file.
## pip_install_from_pyproject

```python
def pip_install_from_pyproject(
    self,
    pyproject_toml: str,
    optional_dependencies: list[str] = [],
    *,
    find_links: Optional[str] = None,  # Passes -f (--find-links) pip install
    index_url: Optional[str] = None,  # Passes -i (--index-url) to pip install
    extra_index_url: Optional[str] = None,  # Passes --extra-index-url to pip install
    pre: bool = False,  # Passes --pre (allow pre-releases) to pip install
    extra_options: str = "",  # Additional options to pass to pip install, e.g. "--no-build-isolation --no-clean"
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    secrets: Sequence[_Secret] = [],
    gpu: GPU_T = None,
) -> "_Image":
```

Install dependencies specified by a local `pyproject.toml` file.

`optional_dependencies` is a list of the keys of the
optional-dependencies section(s) of the `pyproject.toml` file
(e.g. test, doc, experiment, etc). When provided,
all of the packages in each listed section are installed as well.
## uv_pip_install

```python
def uv_pip_install(
    self,
    *packages: Union[str, list[str]],  # A list of Python packages, eg. ["numpy", "matplotlib>=3.5.0"]
    requirements: Optional[list[str]] = None,  # Passes -r (--requirements) to uv pip install
    find_links: Optional[str] = None,  # Passes -f (--find-links) to uv pip install
    index_url: Optional[str] = None,  # Passes -i (--index-url) to uv pip install
    extra_index_url: Optional[str] = None,  # Passes --extra-index-url to uv pip install
    pre: bool = False,  # Allow pre-releases using uv pip install --prerelease allow
    extra_options: str = "",  # Additional options to pass to pip install, e.g. "--no-build-isolation"
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    uv_version: Optional[str] = None,  # uv version to use
    secrets: Sequence[_Secret] = [],
    gpu: GPU_T = None,
) -> "_Image":
```

Install a list of Python packages using uv pip install.

**Examples**

Simple installation:
```python
image = modal.Image.debian_slim().uv_pip_install("torch==2.7.1", "numpy")
```

This method assumes that:
- Python is on the `$PATH` and dependencies are installed with the first Python on the `$PATH`.
- Shell supports backticks for substitution
- `which` command is on the `$PATH`

Added in v1.1.0.
## poetry_install_from_file

```python
def poetry_install_from_file(
    self,
    poetry_pyproject_toml: str,
    poetry_lockfile: Optional[str] = None,  # Path to lockfile. If not provided, uses poetry.lock in same directory.
    *,
    ignore_lockfile: bool = False,  # If set to True, do not use poetry.lock, even when present
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    # Selected optional dependency groups to install (See https://python-poetry.org/docs/cli/#install)
    with_: list[str] = [],
    # Selected optional dependency groups to exclude (See https://python-poetry.org/docs/cli/#install)
    without: list[str] = [],
    only: list[str] = [],  # Only install dependency groups specifed in this list.
    poetry_version: Optional[str] = "latest",  # Version of poetry to install, or None to skip installation
    # If set to True, use old installer. See https://github.com/python-poetry/poetry/issues/3336
    old_installer: bool = False,
    secrets: Sequence[_Secret] = [],
    gpu: GPU_T = None,
) -> "_Image":
```

Install poetry *dependencies* specified by a local `pyproject.toml` file.

If not provided as argument the path to the lockfile is inferred. However, the
file has to exist, unless `ignore_lockfile` is set to `True`.

Note that the root project of the poetry project is not installed, only the dependencies.
For including local python source files see `add_local_python_source`

Poetry will be installed to the Image (using pip) unless `poetry_version` is set to None.
Note that the interpretation of `poetry_version="latest"` depends on the Modal Image Builder
version, with versions 2024.10 and earlier limiting poetry to 1.x.
## uv_sync

```python
def uv_sync(
    self,
    uv_project_dir: str = "./",  # Path to local uv managed project
    *,
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    groups: Optional[list[str]] = None,  # Dependency group to install using `uv sync --group`
    extras: Optional[list[str]] = None,  # Optional dependencies to install using `uv sync --extra`
    frozen: bool = True,  # If True, then we run `uv sync --frozen` when a uv.lock file is present
    extra_options: str = "",  # Extra options to pass to `uv sync`
    uv_version: Optional[str] = None,  # uv version to use
    secrets: Sequence[_Secret] = [],
    gpu: GPU_T = None,
) -> "_Image":
```

Creates a virtual environment with the dependencies in a uv managed project with `uv sync`.

**Examples**
```python
image = modal.Image.debian_slim().uv_sync()
```

The `pyproject.toml` and `uv.lock` in `uv_project_dir` are automatically added to the build context. The
`uv_project_dir` is relative to the current working directory of where `modal` is called.

Added in v1.1.0.
## dockerfile_commands

```python
def dockerfile_commands(
    self,
    *dockerfile_commands: Union[str, list[str]],
    context_files: dict[str, str] = {},
    secrets: Sequence[_Secret] = [],
    gpu: GPU_T = None,
    context_mount: Optional[_Mount] = None,  # Deprecated: the context is now inferred
    context_dir: Optional[Union[Path, str]] = None,  # Context for relative COPY commands
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    ignore: Union[Sequence[str], Callable[[Path], bool]] = AUTO_DOCKERIGNORE,
) -> "_Image":
```

Extend an image with arbitrary Dockerfile-like commands.

**Usage:**

```python
from modal import FilePatternMatcher

# By default a .dockerignore file is used if present in the current working directory
image = modal.Image.debian_slim().dockerfile_commands(
    ["COPY data /data"],
)

image = modal.Image.debian_slim().dockerfile_commands(
    ["COPY data /data"],
    ignore=["*.venv"],
)

image = modal.Image.debian_slim().dockerfile_commands(
    ["COPY data /data"],
    ignore=lambda p: p.is_relative_to(".venv"),
)

image = modal.Image.debian_slim().dockerfile_commands(
    ["COPY data /data"],
    ignore=FilePatternMatcher("**/*.txt"),
)

# When including files is simpler than excluding them, you can use the `~` operator to invert the matcher.
image = modal.Image.debian_slim().dockerfile_commands(
    ["COPY data /data"],
    ignore=~FilePatternMatcher("**/*.py"),
)

# You can also read ignore patterns from a file.
image = modal.Image.debian_slim().dockerfile_commands(
    ["COPY data /data"],
    ignore=FilePatternMatcher.from_file("/path/to/dockerignore"),
)
```
## entrypoint

```python
def entrypoint(
    self,
    entrypoint_commands: list[str],
) -> "_Image":
```

Set the ENTRYPOINT for the image.
## shell

```python
def shell(
    self,
    shell_commands: list[str],
) -> "_Image":
```

Overwrite default shell for the image.
## run_commands

```python
def run_commands(
    self,
    *commands: Union[str, list[str]],
    secrets: Sequence[_Secret] = [],
    gpu: GPU_T = None,
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
) -> "_Image":
```

Extend an image with a list of shell commands to run.
## micromamba

```python
@staticmethod
def micromamba(
    python_version: Optional[str] = None,
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
) -> "_Image":
```

A Micromamba base image. Micromamba allows for fast building of small Conda-based containers.
## micromamba_install

```python
def micromamba_install(
    self,
    # A list of Python packages, eg. ["numpy", "matplotlib>=3.5.0"]
    *packages: Union[str, list[str]],
    # A local path to a file containing package specifications
    spec_file: Optional[str] = None,
    # A list of Conda channels, eg. ["conda-forge", "nvidia"].
    channels: list[str] = [],
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    secrets: Sequence[_Secret] = [],
    gpu: GPU_T = None,
) -> "_Image":
```

Install a list of additional packages using micromamba.
## from_registry

```python
@staticmethod
def from_registry(
    tag: str,
    secret: Optional[_Secret] = None,
    *,
    setup_dockerfile_commands: list[str] = [],
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    add_python: Optional[str] = None,
    **kwargs,
) -> "_Image":
```

Build a Modal Image from a public or private image registry, such as Docker Hub.

The image must be built for the `linux/amd64` platform.

If your image does not come with Python installed, you can use the `add_python` parameter
to specify a version of Python to add to the image. Otherwise, the image is expected to
have Python on PATH as `python`, along with `pip`.

You may also use `setup_dockerfile_commands` to run Dockerfile commands before the
remaining commands run. This might be useful if you want a custom Python installation or to
set a `SHELL`. Prefer `run_commands()` when possible though.

To authenticate against a private registry with static credentials, you must set the `secret` parameter to
a `modal.Secret` containing a username (`REGISTRY_USERNAME`) and
an access token or password (`REGISTRY_PASSWORD`).

To authenticate against private registries with credentials from a cloud provider,
use `Image.from_gcp_artifact_registry()` or `Image.from_aws_ecr()`.

**Examples**

```python
modal.Image.from_registry("python:3.11-slim-bookworm")
modal.Image.from_registry("ubuntu:22.04", add_python="3.11")
modal.Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3")
```
## from_gcp_artifact_registry

```python
@staticmethod
def from_gcp_artifact_registry(
    tag: str,
    secret: Optional[_Secret] = None,
    *,
    setup_dockerfile_commands: list[str] = [],
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    add_python: Optional[str] = None,
    **kwargs,
) -> "_Image":
```

Build a Modal image from a private image in Google Cloud Platform (GCP) Artifact Registry.

You will need to pass a `modal.Secret` containing [your GCP service account key data](https://cloud.google.com/iam/docs/keys-create-delete#creating)
as `SERVICE_ACCOUNT_JSON`. This can be done from the [Secrets](https://modal.com/secrets) page.
Your service account should be granted a specific role depending on the GCP registry used:

- For Artifact Registry images (`pkg.dev` domains) use
  the ["Artifact Registry Reader"](https://cloud.google.com/artifact-registry/docs/access-control#roles) role
- For Container Registry images (`gcr.io` domains) use
  the ["Storage Object Viewer"](https://cloud.google.com/artifact-registry/docs/transition/setup-gcr-repo) role

**Note:** This method does not use `GOOGLE_APPLICATION_CREDENTIALS` as that
variable accepts a path to a JSON file, not the actual JSON string.

See `Image.from_registry()` for information about the other parameters.

**Example**

```python
modal.Image.from_gcp_artifact_registry(
    "us-east1-docker.pkg.dev/my-project-1234/my-repo/my-image:my-version",
    secret=modal.Secret.from_name(
        "my-gcp-secret",
        required_keys=["SERVICE_ACCOUNT_JSON"],
    ),
    add_python="3.11",
)
```
## from_aws_ecr

```python
@staticmethod
def from_aws_ecr(
    tag: str,
    secret: Optional[_Secret] = None,
    *,
    setup_dockerfile_commands: list[str] = [],
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    add_python: Optional[str] = None,
    **kwargs,
) -> "_Image":
```

Build a Modal image from a private image in AWS Elastic Container Registry (ECR).

You will need to pass a `modal.Secret` containing either IAM user credentials or OIDC
configuration to access the target ECR registry.

For IAM user authentication, set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION`.

For OIDC authentication, set `AWS_ROLE_ARN` and `AWS_REGION`.

IAM configuration details can be found in the AWS documentation for
["Private repository policies"](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-policies.html).

For more details on using an AWS role to access ECR, see the [OIDC integration guide](https://modal.com/docs/guide/oidc-integration).

See `Image.from_registry()` for information about the other parameters.

**Example**

```python
modal.Image.from_aws_ecr(
    "000000000000.dkr.ecr.us-east-1.amazonaws.com/my-private-registry:my-version",
    secret=modal.Secret.from_name(
        "aws",
        required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
    ),
    add_python="3.11",
)
```
## from_dockerfile

```python
@staticmethod
def from_dockerfile(
    path: Union[str, Path],  # Filepath to Dockerfile.
    *,
    context_mount: Optional[_Mount] = None,  # Deprecated: the context is now inferred
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    context_dir: Optional[Union[Path, str]] = None,  # Context for relative COPY commands
    secrets: Sequence[_Secret] = [],
    gpu: GPU_T = None,
    add_python: Optional[str] = None,
    build_args: dict[str, str] = {},
    ignore: Union[Sequence[str], Callable[[Path], bool]] = AUTO_DOCKERIGNORE,
) -> "_Image":
```

Build a Modal image from a local Dockerfile.

If your Dockerfile does not have Python installed, you can use the `add_python` parameter
to specify a version of Python to add to the image.

**Usage:**

```python
from modal import FilePatternMatcher

# By default a .dockerignore file is used if present in the current working directory
image = modal.Image.from_dockerfile(
    "./Dockerfile",
    add_python="3.12",
)

image = modal.Image.from_dockerfile(
    "./Dockerfile",
    add_python="3.12",
    ignore=["*.venv"],
)

image = modal.Image.from_dockerfile(
    "./Dockerfile",
    add_python="3.12",
    ignore=lambda p: p.is_relative_to(".venv"),
)

image = modal.Image.from_dockerfile(
    "./Dockerfile",
    add_python="3.12",
    ignore=FilePatternMatcher("**/*.txt"),
)

# When including files is simpler than excluding them, you can use the `~` operator to invert the matcher.
image = modal.Image.from_dockerfile(
    "./Dockerfile",
    add_python="3.12",
    ignore=~FilePatternMatcher("**/*.py"),
)

# You can also read ignore patterns from a file.
image = modal.Image.from_dockerfile(
    "./Dockerfile",
    add_python="3.12",
    ignore=FilePatternMatcher.from_file("/path/to/dockerignore"),
)
```
## debian_slim

```python
@staticmethod
def debian_slim(python_version: Optional[str] = None, force_build: bool = False) -> "_Image":
```

Default image, based on the official `python` Docker images.
## apt_install

```python
def apt_install(
    self,
    *packages: Union[str, list[str]],  # A list of packages, e.g. ["ssh", "libpq-dev"]
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    secrets: Sequence[_Secret] = [],
    gpu: GPU_T = None,
) -> "_Image":
```

Install a list of Debian packages using `apt`.

**Example**

```python
image = modal.Image.debian_slim().apt_install("git")
```
## run_function

```python
def run_function(
    self,
    raw_f: Callable[..., Any],
    *,
    secrets: Sequence[_Secret] = (),  # Optional Modal Secret objects with environment variables for the container
    volumes: dict[Union[str, PurePosixPath], Union[_Volume, _CloudBucketMount]] = {},  # Volume mount paths
    network_file_systems: dict[Union[str, PurePosixPath], _NetworkFileSystem] = {},  # NFS mount paths
    gpu: Union[GPU_T, list[GPU_T]] = None,  # Requested GPU or or list of acceptable GPUs( e.g. ["A10", "A100"])
    cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
    memory: Optional[int] = None,  # How much memory to request, in MiB. This is a soft limit.
    timeout: int = 60 * 60,  # Maximum execution time of the function in seconds.
    cloud: Optional[str] = None,  # Cloud provider to run the function on. Possible values are aws, gcp, oci, auto.
    region: Optional[Union[str, Sequence[str]]] = None,  # Region or regions to run the function on.
    force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    args: Sequence[Any] = (),  # Positional arguments to the function.
    kwargs: dict[str, Any] = {},  # Keyword arguments to the function.
    include_source: bool = True,  # Whether the builder container should have the Function's source added
) -> "_Image":
```

Run user-defined function `raw_f` as an image build step.

The function runs like an ordinary Modal Function, accepting a resource configuration and integrating
with Modal features like Secrets and Volumes. Unlike ordinary Modal Functions, any changes to the
filesystem state will be captured on container exit and saved as a new Image.

**Note**

Only the source code of `raw_f`, the contents of `**kwargs`, and any referenced *global* variables
are used to determine whether the image has changed and needs to be rebuilt.
If this function references other functions or variables, the image will not be rebuilt if you
make changes to them. You can force a rebuild by changing the function's source code itself.

**Example**

```python notest

def my_build_function():
    open("model.pt", "w").write("parameters!")

image = (
    modal.Image
        .debian_slim()
        .pip_install("torch")
        .run_function(my_build_function, secrets=[...], mounts=[...])
)
```
## env

```python
def env(self, vars: dict[str, str]) -> "_Image":
```

Sets the environment variables in an Image.

**Example**

```python
image = (
    modal.Image.debian_slim()
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
```
## workdir

```python
def workdir(self, path: Union[str, PurePosixPath]) -> "_Image":
```

Set the working directory for subsequent image build steps and function execution.

**Example**

```python
image = (
    modal.Image.debian_slim()
    .run_commands("git clone https://xyz app")
    .workdir("/app")
    .run_commands("yarn install")
)
```
## cmd

```python
def cmd(self, cmd: list[str]) -> "_Image":
```

Set the default command (`CMD`) to run when a container is started.

Used with `modal.Sandbox`. Has no effect on `modal.Function`.

**Example**

```python
image = (
    modal.Image.debian_slim().cmd(["python", "app.py"])
)
```
## imports

```python
@contextlib.contextmanager
def imports(self):
```

Used to import packages in global scope that are only available when running remotely.
By using this context manager you can avoid an `ImportError` due to not having certain
packages installed locally.

**Usage:**

```python notest
with image.imports():
    import torch
```

#### NetworkFileSystem

# modal.NetworkFileSystem

```python
class NetworkFileSystem(modal.object.Object)
```

A shared, writable file system accessible by one or more Modal functions.

By attaching this file system as a mount to one or more functions, they can
share and persist data with each other.

**Usage**

```python
import modal

nfs = modal.NetworkFileSystem.from_name("my-nfs", create_if_missing=True)
app = modal.App()

@app.function(network_file_systems={"/root/foo": nfs})
def f():
    pass

@app.function(network_file_systems={"/root/goo": nfs})
def g():
    pass
```

Also see the CLI methods for accessing network file systems:

```
modal nfs --help
```

A `NetworkFileSystem` can also be useful for some local scripting scenarios, e.g.:

```python notest
nfs = modal.NetworkFileSystem.from_name("my-network-file-system")
for chunk in nfs.read_file("my_db_dump.csv"):
    ...
```

## hydrate

```python
def hydrate(self, client: Optional[_Client] = None) -> Self:
```

Synchronize the local object with its identity on the Modal server.

It is rarely necessary to call this method explicitly, as most operations
will lazily hydrate when needed. The main use case is when you need to
access object metadata, such as its ID.

*Added in v0.72.39*: This method replaces the deprecated `.resolve()` method.
## from_name

```python
@staticmethod
def from_name(
    name: str,
    *,
    environment_name: Optional[str] = None,
    create_if_missing: bool = False,
) -> "_NetworkFileSystem":
```

Reference a NetworkFileSystem by its name, creating if necessary.

This is a lazy method that defers hydrating the local object with
metadata from Modal servers until the first time it is actually
used.

```python notest
nfs = NetworkFileSystem.from_name("my-nfs", create_if_missing=True)

@app.function(network_file_systems={"/data": nfs})
def f():
    pass
```
## ephemeral

```python
@classmethod
@contextmanager
def ephemeral(
    cls: type["_NetworkFileSystem"],
    client: Optional[_Client] = None,
    environment_name: Optional[str] = None,
) -> Iterator["_NetworkFileSystem"]:
```

Creates a new ephemeral network filesystem within a context manager:

Usage:
```python
with modal.NetworkFileSystem.ephemeral() as nfs:
    assert nfs.listdir("/") == []
```

```python notest
async with modal.NetworkFileSystem.ephemeral() as nfs:
    assert await nfs.listdir("/") == []
```
## delete

```python
@staticmethod
def delete(name: str, client: Optional[_Client] = None, environment_name: Optional[str] = None):
```

## write_file

```python
@live_method
def write_file(self, remote_path: str, fp: BinaryIO, progress_cb: Optional[Callable[..., Any]] = None) -> int:
```

Write from a file object to a path on the network file system, atomically.

Will create any needed parent directories automatically.

If remote_path ends with `/` it's assumed to be a directory and the
file will be uploaded with its current name to that directory.
## read_file

```python
@live_method_gen
def read_file(self, path: str) -> Iterator[bytes]:
```

Read a file from the network file system
## iterdir

```python
@live_method_gen
def iterdir(self, path: str) -> Iterator[FileEntry]:
```

Iterate over all files in a directory in the network file system.

* Passing a directory path lists all files in the directory (names are relative to the directory)
* Passing a file path returns a list containing only that file's listing description
* Passing a glob path (including at least one * or ** sequence) returns all files matching
that glob path (using absolute paths)
## add_local_file

```python
@live_method
def add_local_file(
    self,
    local_path: Union[Path, str],
    remote_path: Optional[Union[str, PurePosixPath, None]] = None,
    progress_cb: Optional[Callable[..., Any]] = None,
):
```

## add_local_dir

```python
@live_method
def add_local_dir(
    self,
    local_path: Union[Path, str],
    remote_path: Optional[Union[str, PurePosixPath, None]] = None,
    progress_cb: Optional[Callable[..., Any]] = None,
):
```

## listdir

```python
@live_method
def listdir(self, path: str) -> list[FileEntry]:
```

List all files in a directory in the network file system.

* Passing a directory path lists all files in the directory (names are relative to the directory)
* Passing a file path returns a list containing only that file's listing description
* Passing a glob path (including at least one * or ** sequence) returns all files matching
that glob path (using absolute paths)
## remove_file

```python
@live_method
def remove_file(self, path: str, recursive=False):
```

Remove a file in a network file system.

#### Period

# modal.Period

```python
class Period(modal.schedule.Schedule)
```

Create a schedule that runs every given time interval.

**Usage**

```python
import modal
app = modal.App()

@app.function(schedule=modal.Period(days=1))
def f():
    print("This function will run every day")

modal.Period(hours=4)          # runs every 4 hours
modal.Period(minutes=15)       # runs every 15 minutes
modal.Period(seconds=math.pi)  # runs every 3.141592653589793 seconds
```

Only `seconds` can be a float. All other arguments are integers.

Note that `days=1` will trigger the function the same time every day.
This does not have the same behavior as `seconds=84000` since days have
different lengths due to daylight savings and leap seconds. Similarly,
using `months=1` will trigger the function on the same day each month.

This behaves similar to the
[dateutil](https://dateutil.readthedocs.io/en/latest/relativedelta.html)
package.

```python
def __init__(
    self,
    *,
    years: int = 0,
    months: int = 0,
    weeks: int = 0,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: float = 0,
) -> None:
```

#### Proxy

# modal.Proxy

```python
class Proxy(modal.object.Object)
```

Proxy objects give your Modal containers a static outbound IP address.

This can be used for connecting to a remote address with network whitelist, for example
a database. See [the guide](https://modal.com/docs/guide/proxy-ips) for more information.

## hydrate

```python
def hydrate(self, client: Optional[_Client] = None) -> Self:
```

Synchronize the local object with its identity on the Modal server.

It is rarely necessary to call this method explicitly, as most operations
will lazily hydrate when needed. The main use case is when you need to
access object metadata, such as its ID.

*Added in v0.72.39*: This method replaces the deprecated `.resolve()` method.
## from_name

```python
@staticmethod
def from_name(
    name: str,
    *,
    environment_name: Optional[str] = None,
) -> "_Proxy":
```

Reference a Proxy by its name.

In contrast to most other Modal objects, new Proxy objects must be
provisioned via the Dashboard and cannot be created on the fly from code.

#### Queue

# modal.Queue

```python
class Queue(modal.object.Object)
```

Distributed, FIFO queue for data flow in Modal apps.

The queue can contain any object serializable by `cloudpickle`, including Modal objects.

By default, the `Queue` object acts as a single FIFO queue which supports puts and gets (blocking and non-blocking).

**Usage**

```python
from modal import Queue

# Create an ephemeral queue which is anonymous and garbage collected
with Queue.ephemeral() as my_queue:
    # Putting values
    my_queue.put("some value")
    my_queue.put(123)

    # Getting values
    assert my_queue.get() == "some value"
    assert my_queue.get() == 123

    # Using partitions
    my_queue.put(0)
    my_queue.put(1, partition="foo")
    my_queue.put(2, partition="bar")

    # Default and "foo" partition are ignored by the get operation.
    assert my_queue.get(partition="bar") == 2

    # Set custom 10s expiration time on "foo" partition.
    my_queue.put(3, partition="foo", partition_ttl=10)

    # (beta feature) Iterate through items in place (read immutably)
    my_queue.put(1)
    assert [v for v in my_queue.iterate()] == [0, 1]

# You can also create persistent queues that can be used across apps
queue = Queue.from_name("my-persisted-queue", create_if_missing=True)
queue.put(42)
assert queue.get() == 42
```

For more examples, see the [guide](https://modal.com/docs/guide/dicts-and-queues#modal-queues).

**Queue partitions (beta)**

Specifying partition keys gives access to other independent FIFO partitions within the same `Queue` object.
Across any two partitions, puts and gets are completely independent.
For example, a put in one partition does not affect a get in any other partition.

When no partition key is specified (by default), puts and gets will operate on a default partition.
This default partition is also isolated from all other partitions.
Please see the Usage section below for an example using partitions.

**Lifetime of a queue and its partitions**

By default, each partition is cleared 24 hours after the last `put` operation.
A lower TTL can be specified by the `partition_ttl` argument in the `put` or `put_many` methods.
Each partition's expiry is handled independently.

As such, `Queue`s are best used for communication between active functions and not relied on for persistent storage.

On app completion or after stopping an app any associated `Queue` objects are cleaned up.
All its partitions will be cleared.

**Limits**

A single `Queue` can contain up to 100,000 partitions, each with up to 5,000 items. Each item can be up to 1 MiB.

Partition keys must be non-empty and must not exceed 64 bytes.

## hydrate

```python
def hydrate(self, client: Optional[_Client] = None) -> Self:
```

Synchronize the local object with its identity on the Modal server.

It is rarely necessary to call this method explicitly, as most operations
will lazily hydrate when needed. The main use case is when you need to
access object metadata, such as its ID.

*Added in v0.72.39*: This method replaces the deprecated `.resolve()` method.
## objects

```python
class objects(object)
```

Namespace with methods for managing named Queue objects.

### create

```python
@staticmethod
def create(
    name: str,  # Name to use for the new Queue
    *,
    allow_existing: bool = False,  # If True, no-op when the Queue already exists
    environment_name: Optional[str] = None,  # Uses active environment if not specified
    client: Optional[_Client] = None,  # Optional client with Modal credentials
) -> None:
```

Create a new Queue object.

**Examples:**

```python notest
modal.Queue.objects.create("my-queue")
```

Queues will be created in the active environment, or another one can be specified:

```python notest
modal.Queue.objects.create("my-queue", environment_name="dev")
```

By default, an error will be raised if the Queue already exists, but passing
`allow_existing=True` will make the creation attempt a no-op in this case.

```python notest
modal.Queue.objects.create("my-queue", allow_existing=True)
```

Note that this method does not return a local instance of the Queue. You can use
`modal.Queue.from_name` to perform a lookup after creation.

Added in v1.1.2.
### list

```python
@staticmethod
def list(
    *,
    max_objects: Optional[int] = None,  # Limit results to this size
    created_before: Optional[Union[datetime, str]] = None,  # Limit based on creation date
    environment_name: str = "",  # Uses active environment if not specified
    client: Optional[_Client] = None,  # Optional client with Modal credentials
) -> list["_Queue"]:
```

Return a list of hydrated Queue objects.

**Examples:**

```python
queues = modal.Queue.objects.list()
print([q.name for q in queues])
```

Queues will be retreived from the active environment, or another one can be specified:

```python notest
dev_queues = modal.Queue.objects.list(environment_name="dev")
```

By default, all named Queues are returned, newest to oldest. It's also possible to limit the
number of results and to filter by creation date:

```python
queues = modal.Queue.objects.list(max_objects=10, created_before="2025-01-01")
```

Added in v1.1.2.
### delete

```python
@staticmethod
def delete(
    name: str,  # Name of the Queue to delete
    *,
    allow_missing: bool = False,  # If True, don't raise an error if the Queue doesn't exist
    environment_name: Optional[str] = None,  # Uses active environment if not specified
    client: Optional[_Client] = None,  # Optional client with Modal credentials
):
```

Delete a named Queue.

Warning: This deletes an *entire Queue*, not just a specific entry or partition.
Deletion is irreversible and will affect any Apps currently using the Queue.

**Examples:**

```python notest
await modal.Queue.objects.delete("my-queue")
```

Queues will be deleted from the active environment, or another one can be specified:

```python notest
await modal.Queue.objects.delete("my-queue", environment_name="dev")
```

Added in v1.1.2.
## name

```python
@property
def name(self) -> Optional[str]:
```

## validate_partition_key

```python
@staticmethod
def validate_partition_key(partition: Optional[str]) -> bytes:
```

## ephemeral

```python
@classmethod
@contextmanager
def ephemeral(
    cls: type["_Queue"],
    client: Optional[_Client] = None,
    environment_name: Optional[str] = None,
) -> Iterator["_Queue"]:
```

Creates a new ephemeral queue within a context manager:

Usage:
```python
from modal import Queue

with Queue.ephemeral() as q:
    q.put(123)
```

```python notest
async with Queue.ephemeral() as q:
    await q.put.aio(123)
```
## from_name

```python
@staticmethod
def from_name(
    name: str,
    *,
    environment_name: Optional[str] = None,
    create_if_missing: bool = False,
) -> "_Queue":
```

Reference a named Queue, creating if necessary.

This is a lazy method the defers hydrating the local
object with metadata from Modal servers until the first
time it is actually used.

```python
q = modal.Queue.from_name("my-queue", create_if_missing=True)
q.put(123)
```
## info

```python
@live_method
def info(self) -> QueueInfo:
```

Return information about the Queue object.
## clear

```python
@live_method
def clear(self, *, partition: Optional[str] = None, all: bool = False) -> None:
```

Clear the contents of a single partition or all partitions.
## get

```python
@live_method
def get(
    self, block: bool = True, timeout: Optional[float] = None, *, partition: Optional[str] = None
) -> Optional[Any]:
```

Remove and return the next object in the queue.

If `block` is `True` (the default) and the queue is empty, `get` will wait indefinitely for
an object, or until `timeout` if specified. Raises a native `queue.Empty` exception
if the `timeout` is reached.

If `block` is `False`, `get` returns `None` immediately if the queue is empty. The `timeout` is
ignored in this case.
## get_many

```python
@live_method
def get_many(
    self, n_values: int, block: bool = True, timeout: Optional[float] = None, *, partition: Optional[str] = None
) -> list[Any]:
```

Remove and return up to `n_values` objects from the queue.

If there are fewer than `n_values` items in the queue, return all of them.

If `block` is `True` (the default) and the queue is empty, `get` will wait indefinitely for
at least 1 object to be present, or until `timeout` if specified. Raises the stdlib's `queue.Empty`
exception if the `timeout` is reached.

If `block` is `False`, `get` returns `None` immediately if the queue is empty. The `timeout` is
ignored in this case.
## put

```python
@live_method
def put(
    self,
    v: Any,
    block: bool = True,
    timeout: Optional[float] = None,
    *,
    partition: Optional[str] = None,
    partition_ttl: int = 24 * 3600,  # After 24 hours of no activity, this partition will be deletd.
) -> None:
```

Add an object to the end of the queue.

If `block` is `True` and the queue is full, this method will retry indefinitely or
until `timeout` if specified. Raises the stdlib's `queue.Full` exception if the `timeout` is reached.
If blocking it is not recommended to omit the `timeout`, as the operation could wait indefinitely.

If `block` is `False`, this method raises `queue.Full` immediately if the queue is full. The `timeout` is
ignored in this case.
## put_many

```python
@live_method
def put_many(
    self,
    vs: list[Any],
    block: bool = True,
    timeout: Optional[float] = None,
    *,
    partition: Optional[str] = None,
    partition_ttl: int = 24 * 3600,  # After 24 hours of no activity, this partition will be deletd.
) -> None:
```

Add several objects to the end of the queue.

If `block` is `True` and the queue is full, this method will retry indefinitely or
until `timeout` if specified. Raises the stdlib's `queue.Full` exception if the `timeout` is reached.
If blocking it is not recommended to omit the `timeout`, as the operation could wait indefinitely.

If `block` is `False`, this method raises `queue.Full` immediately if the queue is full. The `timeout` is
ignored in this case.
## len

```python
@live_method
def len(self, *, partition: Optional[str] = None, total: bool = False) -> int:
```

Return the number of objects in the queue partition.
## iterate

```python
@warn_if_generator_is_not_consumed()
@live_method_gen
def iterate(
    self, *, partition: Optional[str] = None, item_poll_timeout: float = 0.0
) -> AsyncGenerator[Any, None]:
```

(Beta feature) Iterate through items in the queue without mutation.

Specify `item_poll_timeout` to control how long the iterator should wait for the next time before giving up.

#### Retries

# modal.Retries

```python
class Retries(object)
```

Adds a retry policy to a Modal function.

**Usage**

```python
import modal
app = modal.App()

# Basic configuration.
# This sets a policy of max 4 retries with 1-second delay between failures.
@app.function(retries=4)
def f():
    pass

# Fixed-interval retries with 3-second delay between failures.
@app.function(
    retries=modal.Retries(
        max_retries=2,
        backoff_coefficient=1.0,
        initial_delay=3.0,
    )
)
def g():
    pass

# Exponential backoff, with retry delay doubling after each failure.
@app.function(
    retries=modal.Retries(
        max_retries=4,
        backoff_coefficient=2.0,
        initial_delay=1.0,
    )
)
def h():
    pass
```

```python
def __init__(
    self,
    *,
    # The maximum number of retries that can be made in the presence of failures.
    max_retries: int,
    # Coefficent controlling how much the retry delay increases each retry attempt.
    # A backoff coefficient of 1.0 creates fixed-delay where the delay period always equals the initial delay.
    backoff_coefficient: float = 2.0,
    # Number of seconds that must elapse before the first retry occurs.
    initial_delay: float = 1.0,
    # Maximum length of retry delay in seconds, preventing the delay from growing infinitely.
    max_delay: float = 60.0,
):
```

Construct a new retries policy, supporting exponential and fixed-interval delays via a backoff coefficient.

#### Sandbox

# modal.Sandbox

```python
class Sandbox(modal.object.Object)
```

A `Sandbox` object lets you interact with a running sandbox. This API is similar to Python's
[asyncio.subprocess.Process](https://docs.python.org/3/library/asyncio-subprocess.html#asyncio.subprocess.Process).

Refer to the [guide](https://modal.com/docs/guide/sandbox) on how to spawn and use sandboxes.

## hydrate

```python
def hydrate(self, client: Optional[_Client] = None) -> Self:
```

Synchronize the local object with its identity on the Modal server.

It is rarely necessary to call this method explicitly, as most operations
will lazily hydrate when needed. The main use case is when you need to
access object metadata, such as its ID.

*Added in v0.72.39*: This method replaces the deprecated `.resolve()` method.
## create

```python
@staticmethod
def create(
    *args: str,  # Set the CMD of the Sandbox, overriding any CMD of the container image.
    # Associate the sandbox with an app. Required unless creating from a container.
    app: Optional["modal.app._App"] = None,
    name: Optional[str] = None,  # Optionally give the sandbox a name. Unique within an app.
    image: Optional[_Image] = None,  # The image to run as the container for the sandbox.
    secrets: Sequence[_Secret] = (),  # Environment variables to inject into the sandbox.
    network_file_systems: dict[Union[str, os.PathLike], _NetworkFileSystem] = {},
    timeout: int = 300,  # Maximum lifetime of the sandbox in seconds.
    # The amount of time in seconds that a sandbox can be idle before being terminated.
    idle_timeout: Optional[int] = None,
    workdir: Optional[str] = None,  # Working directory of the sandbox.
    gpu: GPU_T = None,
    cloud: Optional[str] = None,
    region: Optional[Union[str, Sequence[str]]] = None,  # Region or regions to run the sandbox on.
    # Specify, in fractional CPU cores, how many CPU cores to request.
    # Or, pass (request, limit) to additionally specify a hard limit in fractional CPU cores.
    # CPU throttling will prevent a container from exceeding its specified limit.
    cpu: Optional[Union[float, tuple[float, float]]] = None,
    # Specify, in MiB, a memory request which is the minimum memory required.
    # Or, pass (request, limit) to additionally specify a hard limit in MiB.
    memory: Optional[Union[int, tuple[int, int]]] = None,
    block_network: bool = False,  # Whether to block network access
    # List of CIDRs the sandbox is allowed to access. If None, all CIDRs are allowed.
    cidr_allowlist: Optional[Sequence[str]] = None,
    volumes: dict[
        Union[str, os.PathLike], Union[_Volume, _CloudBucketMount]
    ] = {},  # Mount points for Modal Volumes and CloudBucketMounts
    pty_info: Optional[api_pb2.PTYInfo] = None,
    # List of ports to tunnel into the sandbox. Encrypted ports are tunneled with TLS.
    encrypted_ports: Sequence[int] = [],
    # List of encrypted ports to tunnel into the sandbox, using HTTP/2.
    h2_ports: Sequence[int] = [],
    # List of ports to tunnel into the sandbox without encryption.
    unencrypted_ports: Sequence[int] = [],
    # Reference to a Modal Proxy to use in front of this Sandbox.
    proxy: Optional[_Proxy] = None,
    # Enable verbose logging for sandbox operations.
    verbose: bool = False,
    experimental_options: Optional[dict[str, bool]] = None,
    # Enable memory snapshots.
    _experimental_enable_snapshot: bool = False,
    _experimental_scheduler_placement: Optional[
        SchedulerPlacement
    ] = None,  # Experimental controls over fine-grained scheduling (alpha).
    client: Optional[_Client] = None,
    environment_name: Optional[str] = None,  # *DEPRECATED* Optionally override the default environment
) -> "_Sandbox":
```

Create a new Sandbox to run untrusted, arbitrary code.

The Sandbox's corresponding container will be created asynchronously.

**Usage**

```python
app = modal.App.lookup('sandbox-hello-world', create_if_missing=True)
sandbox = modal.Sandbox.create("echo", "hello world", app=app)
print(sandbox.stdout.read())
sandbox.wait()
```
## from_name

```python
@staticmethod
def from_name(
    app_name: str,
    name: str,
    *,
    environment_name: Optional[str] = None,
    client: Optional[_Client] = None,
) -> "_Sandbox":
```

Get a running Sandbox by name from a deployed App.

Raises a modal.exception.NotFoundError if no running sandbox is found with the given name.
A Sandbox's name is the `name` argument passed to `Sandbox.create`.
## from_id

```python
@staticmethod
def from_id(sandbox_id: str, client: Optional[_Client] = None) -> "_Sandbox":
```

Construct a Sandbox from an id and look up the Sandbox result.

The ID of a Sandbox object can be accessed using `.object_id`.
## set_tags

```python
def set_tags(self, tags: dict[str, str], *, client: Optional[_Client] = None):
```

Set tags (key-value pairs) on the Sandbox. Tags can be used to filter results in `Sandbox.list`.
## snapshot_filesystem

```python
def snapshot_filesystem(self, timeout: int = 55) -> _Image:
```

Snapshot the filesystem of the Sandbox.

Returns an [`Image`](https://modal.com/docs/reference/modal.Image) object which
can be used to spawn a new Sandbox with the same filesystem.
## wait

```python
def wait(self, raise_on_termination: bool = True):
```

Wait for the Sandbox to finish running.
## tunnels

```python
def tunnels(self, timeout: int = 50) -> dict[int, Tunnel]:
```

Get Tunnel metadata for the sandbox.

Raises `SandboxTimeoutError` if the tunnels are not available after the timeout.

Returns a dictionary of `Tunnel` objects which are keyed by the container port.

NOTE: Previous to client [v0.64.153](https://modal.com/docs/reference/changelog#064153-2024-09-30), this
returned a list of `TunnelData` objects.
## reload_volumes

```python
def reload_volumes(self) -> None:
```

Reload all Volumes mounted in the Sandbox.

Added in v1.1.0.
## terminate

```python
def terminate(self) -> None:
```

Terminate Sandbox execution.

This is a no-op if the Sandbox has already finished running.
## poll

```python
def poll(self) -> Optional[int]:
```

Check if the Sandbox has finished running.

Returns `None` if the Sandbox is still running, else returns the exit code.
## exec

```python
def exec(
    self,
    *args: str,
    pty_info: Optional[api_pb2.PTYInfo] = None,  # Deprecated: internal use only
    stdout: StreamType = StreamType.PIPE,
    stderr: StreamType = StreamType.PIPE,
    timeout: Optional[int] = None,
    workdir: Optional[str] = None,
    secrets: Sequence[_Secret] = (),
    # Encode output as text.
    text: bool = True,
    # Control line-buffered output.
    # -1 means unbuffered, 1 means line-buffered (only available if `text=True`).
    bufsize: Literal[-1, 1] = -1,
    # Internal option to set terminal size and metadata
    _pty_info: Optional[api_pb2.PTYInfo] = None,
):
```

Execute a command in the Sandbox and return a ContainerProcess handle.

See the [`ContainerProcess`](https://modal.com/docs/reference/modal.container_process#modalcontainer_processcontainerprocess)
docs for more information.

**Usage**

```python
app = modal.App.lookup("my-app", create_if_missing=True)

sandbox = modal.Sandbox.create("sleep", "infinity", app=app)

process = sandbox.exec("bash", "-c", "for i in $(seq 1 10); do echo foo $i; sleep 0.5; done")

for line in process.stdout:
    print(line)
```
## open

```python
def open(
    self,
    path: str,
    mode: Union["_typeshed.OpenTextMode", "_typeshed.OpenBinaryMode"] = "r",
):
```

[Alpha] Open a file in the Sandbox and return a FileIO handle.

See the [`FileIO`](https://modal.com/docs/reference/modal.file_io#modalfile_iofileio) docs for more information.

**Usage**

```python notest
sb = modal.Sandbox.create(app=sb_app)
f = sb.open("/test.txt", "w")
f.write("hello")
f.close()
```
## ls

```python
def ls(self, path: str) -> list[str]:
```

[Alpha] List the contents of a directory in the Sandbox.
## mkdir

```python
def mkdir(self, path: str, parents: bool = False) -> None:
```

[Alpha] Create a new directory in the Sandbox.
## rm

```python
def rm(self, path: str, recursive: bool = False) -> None:
```

[Alpha] Remove a file or directory in the Sandbox.
## watch

```python
def watch(
    self,
    path: str,
    filter: Optional[list[FileWatchEventType]] = None,
    recursive: Optional[bool] = None,
    timeout: Optional[int] = None,
) -> Iterator[FileWatchEvent]:
```

[Alpha] Watch a file or directory in the Sandbox for changes.
## stdout

```python
@property
def stdout(self) -> _StreamReader[str]:
```

[`StreamReader`](https://modal.com/docs/reference/modal.io_streams#modalio_streamsstreamreader) for
the sandbox's stdout stream.
## stderr

```python
@property
def stderr(self) -> _StreamReader[str]:
```

[`StreamReader`](https://modal.com/docs/reference/modal.io_streams#modalio_streamsstreamreader) for
the Sandbox's stderr stream.
## stdin

```python
@property
def stdin(self) -> _StreamWriter:
```

[`StreamWriter`](https://modal.com/docs/reference/modal.io_streams#modalio_streamsstreamwriter) for
the Sandbox's stdin stream.
## returncode

```python
@property
def returncode(self) -> Optional[int]:
```

Return code of the Sandbox process if it has finished running, else `None`.
## list

```python
@staticmethod
def list(
    *, app_id: Optional[str] = None, tags: Optional[dict[str, str]] = None, client: Optional[_Client] = None
) -> AsyncGenerator["_Sandbox", None]:
```

List all Sandboxes for the current Environment or App ID (if specified). If tags are specified, only
Sandboxes that have at least those tags are returned. Returns an iterator over `Sandbox` objects.

#### SandboxSnapshot

# modal.SandboxSnapshot

```python
class SandboxSnapshot(modal.object.Object)
```

> Sandbox memory snapshots are in **early preview**.

A `SandboxSnapshot` object lets you interact with a stored Sandbox snapshot that was created by calling
`._experimental_snapshot()` on a Sandbox instance. This includes both the filesystem and memory state of
the original Sandbox at the time the snapshot was taken.

## hydrate

```python
def hydrate(self, client: Optional[_Client] = None) -> Self:
```

Synchronize the local object with its identity on the Modal server.

It is rarely necessary to call this method explicitly, as most operations
will lazily hydrate when needed. The main use case is when you need to
access object metadata, such as its ID.

*Added in v0.72.39*: This method replaces the deprecated `.resolve()` method.
## from_id

```python
@staticmethod
def from_id(sandbox_snapshot_id: str, client: Optional[_Client] = None):
```

Construct a `SandboxSnapshot` object from a sandbox snapshot ID.

#### Secret

# modal.Secret

```python
class Secret(modal.object.Object)
```

Secrets provide a dictionary of environment variables for images.

Secrets are a secure way to add credentials and other sensitive information
to the containers your functions run in. You can create and edit secrets on
[the dashboard](https://modal.com/secrets), or programmatically from Python code.

See [the secrets guide page](https://modal.com/docs/guide/secrets) for more information.

## hydrate

```python
def hydrate(self, client: Optional[_Client] = None) -> Self:
```

Synchronize the local object with its identity on the Modal server.

It is rarely necessary to call this method explicitly, as most operations
will lazily hydrate when needed. The main use case is when you need to
access object metadata, such as its ID.

*Added in v0.72.39*: This method replaces the deprecated `.resolve()` method.
## objects

```python
class objects(object)
```

Namespace with methods for managing named Secret objects.

### create

```python
@staticmethod
def create(
    name: str,  # Name to use for the new Secret
    env_dict: dict[str, str],  # Key-value pairs to set in the Secret
    *,
    allow_existing: bool = False,  # If True, no-op when the Secret already exists
    environment_name: Optional[str] = None,  # Uses active environment if not specified
    client: Optional[_Client] = None,  # Optional client with Modal credentials
) -> None:
```

Create a new Secret object.

**Examples:**

```python notest
contents = {"MY_KEY": "my-value", "MY_OTHER_KEY": "my-other-value"}
modal.Secret.objects.create("my-secret", contents)
```

Secrets will be created in the active environment, or another one can be specified:

```python notest
modal.Secret.objects.create("my-secret", contents, environment_name="dev")
```

By default, an error will be raised if the Secret already exists, but passing
`allow_existing=True` will make the creation attempt a no-op in this case.
If the `env_dict` data differs from the existing Secret, it will be ignored.

```python notest
modal.Secret.objects.create("my-secret", contents, allow_existing=True)
```

Note that this method does not return a local instance of the Secret. You can use
`modal.Secret.from_name` to perform a lookup after creation.

Added in v1.1.2.
### list

```python
@staticmethod
def list(
    *,
    max_objects: Optional[int] = None,  # Limit requests to this size
    created_before: Optional[Union[datetime, str]] = None,  # Limit based on creation date
    environment_name: str = "",  # Uses active environment if not specified
    client: Optional[_Client] = None,  # Optional client with Modal credentials
) -> list["_Secret"]:
```

Return a list of hydrated Secret objects.

**Examples:**

```python
secrets = modal.Secret.objects.list()
print([s.name for s in secrets])
```

Secrets will be retreived from the active environment, or another one can be specified:

```python notest
dev_secrets = modal.Secret.objects.list(environment_name="dev")
```

By default, all named Secrets are returned, newest to oldest. It's also possible to limit the
number of results and to filter by creation date:

```python
secrets = modal.Secret.objects.list(max_objects=10, created_before="2025-01-01")
```

Added in v1.1.2.
### delete

```python
@staticmethod
def delete(
    name: str,  # Name of the Secret to delete
    *,
    allow_missing: bool = False,  # If True, don't raise an error if the Secret doesn't exist
    environment_name: Optional[str] = None,  # Uses active environment if not specified
    client: Optional[_Client] = None,  # Optional client with Modal credentials
):
```

Delete a named Secret.

Warning: Deletion is irreversible and will affect any Apps currently using the Secret.

**Examples:**

```python notest
await modal.Secret.objects.delete("my-secret")
```

Secrets will be deleted from the active environment, or another one can be specified:

```python notest
await modal.Secret.objects.delete("my-secret", environment_name="dev")
```

Added in v1.1.2.
## name

```python
@property
def name(self) -> Optional[str]:
```

## from_dict

```python
@staticmethod
def from_dict(
    env_dict: dict[
        str, Union[str, None]
    ] = {},  # dict of entries to be inserted as environment variables in functions using the secret
) -> "_Secret":
```

Create a secret from a str-str dictionary. Values can also be `None`, which is ignored.

Usage:
```python
@app.function(secrets=[modal.Secret.from_dict({"FOO": "bar"})])
def run():
    print(os.environ["FOO"])
```
## from_local_environ

```python
@staticmethod
def from_local_environ(
    env_keys: list[str],  # list of local env vars to be included for remote execution
) -> "_Secret":
```

Create secrets from local environment variables automatically.
## from_dotenv

```python
@staticmethod
def from_dotenv(path=None, *, filename=".env") -> "_Secret":
```

Create secrets from a .env file automatically.

If no argument is provided, it will use the current working directory as the starting
point for finding a `.env` file. Note that it does not use the location of the module
calling `Secret.from_dotenv`.

If called with an argument, it will use that as a starting point for finding `.env` files.
In particular, you can call it like this:
```python
@app.function(secrets=[modal.Secret.from_dotenv(__file__)])
def run():
    print(os.environ["USERNAME"])  # Assumes USERNAME is defined in your .env file
```

This will use the location of the script calling `modal.Secret.from_dotenv` as a
starting point for finding the `.env` file.

A file named `.env` is expected by default, but this can be overridden with the `filename`
keyword argument:

```python
@app.function(secrets=[modal.Secret.from_dotenv(filename=".env-dev")])
def run():
    ...
```
## from_name

```python
@staticmethod
def from_name(
    name: str,
    *,
    environment_name: Optional[str] = None,
    required_keys: list[
        str
    ] = [],  # Optionally, a list of required environment variables (will be asserted server-side)
) -> "_Secret":
```

Reference a Secret by its name.

In contrast to most other Modal objects, named Secrets must be provisioned
from the Dashboard. See other methods for alternate ways of creating a new
Secret from code.

```python
secret = modal.Secret.from_name("my-secret")

@app.function(secrets=[secret])
def run():
   ...
```
## info

```python
@live_method
def info(self) -> SecretInfo:
```

Return information about the Secret object.

#### Tunnel

# modal.Tunnel

```python
class Tunnel(object)
```

A port forwarded from within a running Modal container. Created by `modal.forward()`.

**Important:** This is an experimental API which may change in the future.

```python
def __init__(self, host: str, port: int, unencrypted_host: str, unencrypted_port: int) -> None
```

## url

```python
@property
def url(self) -> str:
```

Get the public HTTPS URL of the forwarded port.
## tls_socket

```python
@property
def tls_socket(self) -> tuple[str, int]:
```

Get the public TLS socket as a (host, port) tuple.
## tcp_socket

```python
@property
def tcp_socket(self) -> tuple[str, int]:
```

Get the public TCP socket as a (host, port) tuple.

#### Volume

# modal.Volume

```python
class Volume(modal.object.Object)
```

A writeable volume that can be used to share files between one or more Modal functions.

The contents of a volume is exposed as a filesystem. You can use it to share data between different functions, or
to persist durable state across several instances of the same function.

Unlike a networked filesystem, you need to explicitly reload the volume to see changes made since it was mounted.
Similarly, you need to explicitly commit any changes you make to the volume for the changes to become visible
outside the current container.

Concurrent modification is supported, but concurrent modifications of the same files should be avoided! Last write
wins in case of concurrent modification of the same file - any data the last writer didn't have when committing
changes will be lost!

As a result, volumes are typically not a good fit for use cases where you need to make concurrent modifications to
the same file (nor is distributed file locking supported).

Volumes can only be reloaded if there are no open files for the volume - attempting to reload with open files
will result in an error.

**Usage**

```python
import modal

app = modal.App()
volume = modal.Volume.from_name("my-persisted-volume", create_if_missing=True)

@app.function(volumes={"/root/foo": volume})
def f():
    with open("/root/foo/bar.txt", "w") as f:
        f.write("hello")
    volume.commit()  # Persist changes

@app.function(volumes={"/root/foo": volume})
def g():
    volume.reload()  # Fetch latest changes
    with open("/root/foo/bar.txt", "r") as f:
        print(f.read())
```

## hydrate

```python
def hydrate(self, client: Optional[_Client] = None) -> Self:
```

Synchronize the local object with its identity on the Modal server.

It is rarely necessary to call this method explicitly, as most operations
will lazily hydrate when needed. The main use case is when you need to
access object metadata, such as its ID.

*Added in v0.72.39*: This method replaces the deprecated `.resolve()` method.
## objects

```python
class objects(object)
```

Namespace with methods for managing named Volume objects.

### create

```python
@staticmethod
def create(
    name: str,  # Name to use for the new Volume
    *,
    version: Optional[int] = None,  # Experimental: Configure the backend VolumeFS version
    allow_existing: bool = False,  # If True, no-op when the Volume already exists
    environment_name: Optional[str] = None,  # Uses active environment if not specified
    client: Optional[_Client] = None,  # Optional client with Modal credentials
) -> None:
```

Create a new Volume object.

**Examples:**

```python notest
modal.Volume.objects.create("my-volume")
```

Volumes will be created in the active environment, or another one can be specified:

```python notest
modal.Volume.objects.create("my-volume", environment_name="dev")
```

By default, an error will be raised if the Volume already exists, but passing
`allow_existing=True` will make the creation attempt a no-op in this case.

```python notest
modal.Volume.objects.create("my-volume", allow_existing=True)
```

Note that this method does not return a local instance of the Volume. You can use
`modal.Volume.from_name` to perform a lookup after creation.

Added in v1.1.2.
### list

```python
@staticmethod
def list(
    *,
    max_objects: Optional[int] = None,  # Limit requests to this size
    created_before: Optional[Union[datetime, str]] = None,  # Limit based on creation date
    environment_name: str = "",  # Uses active environment if not specified
    client: Optional[_Client] = None,  # Optional client with Modal credentials
) -> list["_Volume"]:
```

Return a list of hydrated Volume objects.

**Examples:**

```python
volumes = modal.Volume.objects.list()
print([v.name for v in volumes])
```

Volumes will be retreived from the active environment, or another one can be specified:

```python notest
dev_volumes = modal.Volume.objects.list(environment_name="dev")
```

By default, all named Volumes are returned, newest to oldest. It's also possible to limit the
number of results and to filter by creation date:

```python
volumes = modal.Volume.objects.list(max_objects=10, created_before="2025-01-01")
```

Added in v1.1.2.
### delete

```python
@staticmethod
def delete(
    name: str,  # Name of the Volume to delete
    *,
    allow_missing: bool = False,  # If True, don't raise an error if the Volume doesn't exist
    environment_name: Optional[str] = None,  # Uses active environment if not specified
    client: Optional[_Client] = None,  # Optional client with Modal credentials
):
```

Delete a named Volume.

Warning: This deletes an *entire Volume*, not just a specific file.
Deletion is irreversible and will affect any Apps currently using the Volume.

**Examples:**

```python notest
await modal.Volume.objects.delete("my-volume")
```

Volumes will be deleted from the active environment, or another one can be specified:

```python notest
await modal.Volume.objects.delete("my-volume", environment_name="dev")
```

Added in v1.1.2.
## name

```python
@property
def name(self) -> Optional[str]:
```

## read_only

```python
def read_only(self) -> "_Volume":
```

Configure Volume to mount as read-only.

**Example**

```python
import modal

volume = modal.Volume.from_name("my-volume", create_if_missing=True)

@app.function(volumes={"/mnt/items": volume.read_only()})
def f():
    with open("/mnt/items/my-file.txt") as f:
        return f.read()
```

The Volume is mounted as a read-only volume in a function. Any file system write operation into the
mounted volume will result in an error.

Added in v1.0.5.
## from_name

```python
@staticmethod
def from_name(
    name: str,
    *,
    environment_name: Optional[str] = None,
    create_if_missing: bool = False,
    version: "typing.Optional[modal_proto.api_pb2.VolumeFsVersion.ValueType]" = None,
) -> "_Volume":
```

Reference a Volume by name, creating if necessary.

This is a lazy method that defers hydrating the local
object with metadata from Modal servers until the first
time is is actually used.

```python
vol = modal.Volume.from_name("my-volume", create_if_missing=True)

app = modal.App()

# Volume refers to the same object, even across instances of `app`.
@app.function(volumes={"/data": vol})
def f():
    pass
```
## ephemeral

```python
@classmethod
@contextmanager
def ephemeral(
    cls: type["_Volume"],
    client: Optional[_Client] = None,
    environment_name: Optional[str] = None,
    version: "typing.Optional[modal_proto.api_pb2.VolumeFsVersion.ValueType]" = None,
) -> AsyncGenerator["_Volume", None]:
```

Creates a new ephemeral volume within a context manager:

Usage:
```python
import modal
with modal.Volume.ephemeral() as vol:
    assert vol.listdir("/") == []
```

```python notest
async with modal.Volume.ephemeral() as vol:
    assert await vol.listdir("/") == []
```
## info

```python
@live_method
def info(self) -> VolumeInfo:
```

Return information about the Volume object.
## commit

```python
@live_method
def commit(self):
```

Commit changes to a mounted volume.

If successful, the changes made are now persisted in durable storage and available to other containers accessing
the volume.
## reload

```python
@live_method
def reload(self):
```

Make latest committed state of volume available in the running container.

Any uncommitted changes to the volume, such as new or modified files, may implicitly be committed when
reloading.

Reloading will fail if there are open files for the volume.
## iterdir

```python
@live_method_gen
def iterdir(self, path: str, *, recursive: bool = True) -> Iterator[FileEntry]:
```

Iterate over all files in a directory in the volume.

Passing a directory path lists all files in the directory. For a file path, return only that
file's description. If `recursive` is set to True, list all files and folders under the path
recursively.
## listdir

```python
@live_method
def listdir(self, path: str, *, recursive: bool = False) -> list[FileEntry]:
```

List all files under a path prefix in the modal.Volume.

Passing a directory path lists all files in the directory. For a file path, return only that
file's description. If `recursive` is set to True, list all files and folders under the path
recursively.
## read_file

```python
@live_method_gen
def read_file(self, path: str) -> Iterator[bytes]:
```

Read a file from the modal.Volume.

Note - this function is primarily intended to be used outside of a Modal App.
For more information on downloading files from a Modal Volume, see
[the guide](https://modal.com/docs/guide/volumes).

**Example:**

```python notest
vol = modal.Volume.from_name("my-modal-volume")
data = b""
for chunk in vol.read_file("1mb.csv"):
    data += chunk
print(len(data))  # == 1024 * 1024
```
## remove_file

```python
@live_method
def remove_file(self, path: str, recursive: bool = False) -> None:
```

Remove a file or directory from a volume.
## copy_files

```python
@live_method
def copy_files(self, src_paths: Sequence[str], dst_path: str, recursive: bool = False) -> None:
```

Copy files within the volume from src_paths to dst_path.
The semantics of the copy operation follow those of the UNIX cp command.

The `src_paths` parameter is a list. If you want to copy a single file, you should pass a list with a
single element.

`src_paths` and `dst_path` should refer to the desired location *inside* the volume. You do not need to prepend
the volume mount path.

**Usage**

```python notest
vol = modal.Volume.from_name("my-modal-volume")

vol.copy_files(["bar/example.txt"], "bar2")  # Copy files to another directory
vol.copy_files(["bar/example.txt"], "bar/example2.txt")  # Rename a file by copying
```

Note that if the volume is already mounted on the Modal function, you should use normal filesystem operations
like `os.rename()` and then `commit()` the volume. The `copy_files()` method is useful when you don't have
the volume mounted as a filesystem, e.g. when running a script on your local computer.
## batch_upload

```python
@live_method
def batch_upload(self, force: bool = False) -> "_AbstractVolumeUploadContextManager":
```

Initiate a batched upload to a volume.

To allow overwriting existing files, set `force` to `True` (you cannot overwrite existing directories with
uploaded files regardless).

**Example:**

```python notest
vol = modal.Volume.from_name("my-modal-volume")

with vol.batch_upload() as batch:
    batch.put_file("local-path.txt", "/remote-path.txt")
    batch.put_directory("/local/directory/", "/remote/directory")
    batch.put_file(io.BytesIO(b"some data"), "/foobar")
```
## rename

```python
@staticmethod
def rename(
    old_name: str,
    new_name: str,
    *,
    client: Optional[_Client] = None,
    environment_name: Optional[str] = None,
):
```

#### asgi app

# modal.asgi_app

```python
def asgi_app(
    *,
    label: Optional[str] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
    custom_domains: Optional[Iterable[str]] = None,  # Deploy this endpoint on a custom domain.
    requires_proxy_auth: bool = False,  # Require Modal-Key and Modal-Secret HTTP Headers on requests.
) -> Callable[[Union[_PartialFunction, NullaryFuncOrMethod]], _PartialFunction]:
```

Decorator for registering an ASGI app with a Modal function.

Asynchronous Server Gateway Interface (ASGI) is a standard for Python
synchronous and asynchronous apps, supported by all popular Python web
libraries. This is an advanced decorator that gives full flexibility in
defining one or more web endpoints on Modal.

**Usage:**

```python
from typing import Callable

@app.function()
@modal.asgi_app()
def create_asgi() -> Callable:
    ...
```

To learn how to use Modal with popular web frameworks, see the
[guide on web endpoints](https://modal.com/docs/guide/webhooks).

#### batched

# modal.batched

```python
def batched(
    *,
    max_batch_size: int,
    wait_ms: int,
) -> Callable[
    [Union[_PartialFunction[P, ReturnType, ReturnType], Callable[P, ReturnType]]],
    _PartialFunction[P, ReturnType, ReturnType],
]:
```

Decorator for functions or class methods that should be batched.

**Usage**

```python
# Stack the decorator under `@app.function()` to enable dynamic batching
@app.function()
@modal.batched(max_batch_size=4, wait_ms=1000)
async def batched_multiply(xs: list[int], ys: list[int]) -> list[int]:
    return [x * y for x, y in zip(xs, ys)]

# call batched_multiply with individual inputs
# batched_multiply.remote.aio(2, 100)

# With `@app.cls()`, apply the decorator to a method (this may change in the future)
@app.cls()
class BatchedClass:
    @modal.batched(max_batch_size=4, wait_ms=1000)
    def batched_multiply(self, xs: list[int], ys: list[int]) -> list[int]:
        return [x * y for x, y in zip(xs, ys)]
```

See the [dynamic batching guide](https://modal.com/docs/guide/dynamic-batching) for more information.

#### call graph

# modal.call_graph

## modal.call_graph.InputInfo

```python
class InputInfo(object)
```

Simple data structure storing information about a function input.

```python
def __init__(self, input_id: str, function_call_id: str, task_id: str, status: modal.call_graph.InputStatus, function_name: str, module_name: str, children: list['InputInfo']) -> None
```

## modal.call_graph.InputStatus

```python
class InputStatus(enum.IntEnum)
```

Enum representing status of a function input.

The possible values are:

* `PENDING`
* `SUCCESS`
* `FAILURE`
* `INIT_FAILURE`
* `TERMINATED`
* `TIMEOUT`

#### concurrent

# modal.concurrent

```python
def concurrent(
    *,
    max_inputs: int,  # Hard limit on each container's input concurrency
    target_inputs: Optional[int] = None,  # Input concurrency that Modal's autoscaler should target
) -> Callable[
    [Union[Callable[P, ReturnType], _PartialFunction[P, ReturnType, ReturnType]]],
    _PartialFunction[P, ReturnType, ReturnType],
]:
```

Decorator that allows individual containers to handle multiple inputs concurrently.

The concurrency mechanism depends on whether the function is async or not:
- Async functions will run inputs on a single thread as asyncio tasks.
- Synchronous functions will use multi-threading. The code must be thread-safe.

Input concurrency will be most useful for workflows that are IO-bound
(e.g., making network requests) or when running an inference server that supports
dynamic batching.

When `target_inputs` is set, Modal's autoscaler will try to provision resources
such that each container is running that many inputs concurrently, rather than
autoscaling based on `max_inputs`. Containers may burst up to up to `max_inputs`
if resources are insufficient to remain at the target concurrency, e.g. when the
arrival rate of inputs increases. This can trade-off a small increase in average
latency to avoid larger tail latencies from input queuing.

**Examples:**
```python
# Stack the decorator under `@app.function()` to enable input concurrency
@app.function()
@modal.concurrent(max_inputs=100)
async def f(data):
    # Async function; will be scheduled as asyncio task
    ...

# With `@app.cls()`, apply the decorator at the class level, not on individual methods
@app.cls()
@modal.concurrent(max_inputs=100, target_inputs=80)
class C:
    @modal.method()
    def f(self, data):
        # Sync function; must be thread-safe
        ...

```

*Added in v0.73.148:* This decorator replaces the `allow_concurrent_inputs` parameter
in `@app.function()` and `@app.cls()`.

#### config

# modal.config

Modal intentionally keeps configurability to a minimum.

The main configuration options are the API tokens: the token id and the token secret.
These can be configured in two ways:

1. By running the `modal token set` command.
   This writes the tokens to `.modal.toml` file in your home directory.
2. By setting the environment variables `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.
   This takes precedence over the previous method.

.modal.toml
---------------

The `.modal.toml` file is generally stored in your home directory.
It should look like this::

```toml
[default]
token_id = "ak-12345..."
token_secret = "as-12345..."
```

You can create this file manually, or you can run the `modal token set ...`
command (see below).

Setting tokens using the CLI
----------------------------

You can set a token by running the command::

```
modal token set \
  --token-id <token id> \
  --token-secret <token secret>
```

This will write the token id and secret to `.modal.toml`.

If the token id or secret is provided as the string `-` (a single dash),
then it will be read in a secret way from stdin instead.

Other configuration options
---------------------------

Other possible configuration options are:

* `loglevel` (in the .toml file) / `MODAL_LOGLEVEL` (as an env var).
  Defaults to `WARNING`. Set this to `DEBUG` to see internal messages.
* `logs_timeout` (in the .toml file) / `MODAL_LOGS_TIMEOUT` (as an env var).
  Defaults to 10.
  Number of seconds to wait for logs to drain when closing the session,
  before giving up.
* `force_build` (in the .toml file) / `MODAL_FORCE_BUILD` (as an env var).
  Defaults to False.
  When set, ignores the Image cache and builds all Image layers. Note that this
  will break the cache for all images based on the rebuilt layers, so other images
  may rebuild on subsequent runs / deploys even if the config is reverted.
* `ignore_cache` (in the .toml file) / `MODAL_IGNORE_CACHE` (as an env var).
  Defaults to False.
  When set, ignores the Image cache and builds all Image layers. Unlike `force_build`,
  this will not overwrite the cache for other images that have the same recipe.
  Subsequent runs that do not use this option will pull the *previous* Image from
  the cache, if one exists. It can be useful for testing an App's robustness to
  Image rebuilds without clobbering Images used by other Apps.
* `traceback` (in the .toml file) / `MODAL_TRACEBACK` (as an env var).
  Defaults to False. Enables printing full tracebacks on unexpected CLI
  errors, which can be useful for debugging client issues.
* `log_pattern` (in the .toml file) / MODAL_LOG_PATTERN` (as an env var).
  Defaults to "[modal-client] %(asctime)s %(message)s"
  The log formatting pattern that will be used by the modal client itself.
  See https://docs.python.org/3/library/logging.html#logrecord-attributes for available
  log attributes.

Meta-configuration
------------------

Some "meta-options" are set using environment variables only:

* `MODAL_CONFIG_PATH` lets you override the location of the .toml file,
  by default `~/.modal.toml`.
* `MODAL_PROFILE` lets you use multiple sections in the .toml file
  and switch between them. It defaults to "default".

## modal.config.Config

```python
class Config(object)
```

Singleton that holds configuration used by Modal internally.

```python
def __init__(self):
```

### get

```python
def get(self, key, profile=None, use_env=True):
```

Looks up a configuration value.

Will check (in decreasing order of priority):
1. Any environment variable of the form MODAL_FOO_BAR (when use_env is True)
2. Settings in the user's .toml configuration file
3. The default value of the setting
### override_locally

```python
def override_locally(self, key: str, value: str):
    # Override setting in this process by overriding environment variable for the setting
    #
    # Does NOT write back to settings file etc.
```

### to_dict

```python
def to_dict(self):
```

## modal.config.config_profiles

```python
def config_profiles():
```

List the available modal profiles in the .modal.toml file.
## modal.config.config_set_active_profile

```python
def config_set_active_profile(profile: str) -> None:
```

Set the user's active modal profile by writing it to the `.modal.toml` file.

#### container process

# modal.container_process

## modal.container_process.ContainerProcess

```python
class ContainerProcess(typing.Generic)
```

```python
def __init__(
    self,
    process_id: str,
    client: _Client,
    stdout: StreamType = StreamType.PIPE,
    stderr: StreamType = StreamType.PIPE,
    exec_deadline: Optional[float] = None,
    text: bool = True,
    by_line: bool = False,
) -> None:
```

### stdout

```python
@property
def stdout(self) -> _StreamReader[T]:
```

StreamReader for the container process's stdout stream.
### stderr

```python
@property
def stderr(self) -> _StreamReader[T]:
```

StreamReader for the container process's stderr stream.
### stdin

```python
@property
def stdin(self) -> _StreamWriter:
```

StreamWriter for the container process's stdin stream.
### returncode

```python
@property
def returncode(self) -> int:
```

### poll

```python
def poll(self) -> Optional[int]:
```

Check if the container process has finished running.

Returns `None` if the process is still running, else returns the exit code.
### wait

```python
def wait(self) -> int:
```

Wait for the container process to finish running. Returns the exit code.

#### current function call id

# modal.current_function_call_id

```python
def current_function_call_id() -> Optional[str]:
```

Returns the function call ID for the current input.

Can only be called from Modal function (i.e. in a container context).

```python
from modal import current_function_call_id

@app.function()
def process_stuff():
    print(f"Starting to process input from {current_function_call_id()}")
```

#### current input id

# modal.current_input_id

```python
def current_input_id() -> Optional[str]:
```

Returns the input ID for the current input.

Can only be called from Modal function (i.e. in a container context).

```python
from modal import current_input_id

@app.function()
def process_stuff():
    print(f"Starting to process {current_input_id()}")
```

#### enable output

# modal.enable_output

```python
@contextlib.contextmanager
def enable_output(show_progress: bool = True) -> Generator[None, None, None]:
```

Context manager that enable output when using the Python SDK.

This will print to stdout and stderr things such as
1. Logs from running functions
2. Status of creating objects
3. Map progress

Example:
```python
app = modal.App()
with modal.enable_output():
    with app.run():
        ...
```

#### enter

# modal.enter

```python
def enter(
    *,
    snap: bool = False,
) -> Callable[[Union[_PartialFunction, NullaryMethod]], _PartialFunction]:
```

Decorator for methods which should be executed when a new container is started.

See the [lifeycle function guide](https://modal.com/docs/guide/lifecycle-functions#enter) for more information.

#### exception

# modal.exception

## modal.exception.AlreadyExistsError

```python
class AlreadyExistsError(modal.exception.Error)
```

Raised when a resource creation conflicts with an existing resource.

## modal.exception.AuthError

```python
class AuthError(modal.exception.Error)
```

Raised when a client has missing or invalid authentication.

## modal.exception.ClientClosed

```python
class ClientClosed(modal.exception.Error)
```

## modal.exception.ConnectionError

```python
class ConnectionError(modal.exception.Error)
```

Raised when an issue occurs while connecting to the Modal servers.

## modal.exception.DeprecationError

```python
class DeprecationError(UserWarning)
```

UserWarning category emitted when a deprecated Modal feature or API is used.

## modal.exception.DeserializationError

```python
class DeserializationError(modal.exception.Error)
```

Raised to provide more context when an error is encountered during deserialization.

## modal.exception.ExecutionError

```python
class ExecutionError(modal.exception.Error)
```

Raised when something unexpected happened during runtime.

## modal.exception.FilesystemExecutionError

```python
class FilesystemExecutionError(modal.exception.Error)
```

Raised when an unknown error is thrown during a container filesystem operation.

## modal.exception.FunctionTimeoutError

```python
class FunctionTimeoutError(modal.exception.TimeoutError)
```

Raised when a Function exceeds its execution duration limit and times out.

## modal.exception.InputCancellation

```python
class InputCancellation(BaseException)
```

Raised when the current input is cancelled by the task

Intentionally a BaseException instead of an Exception, so it won't get
caught by unspecified user exception clauses that might be used for retries and
other control flow.

## modal.exception.InteractiveTimeoutError

```python
class InteractiveTimeoutError(modal.exception.TimeoutError)
```

Raised when interactive frontends time out while trying to connect to a container.

## modal.exception.InternalFailure

```python
class InternalFailure(modal.exception.Error)
```

Retriable internal error.

## modal.exception.InvalidError

```python
class InvalidError(modal.exception.Error)
```

Raised when user does something invalid.

## modal.exception.ModuleNotMountable

```python
class ModuleNotMountable(Exception)
```

## modal.exception.MountUploadTimeoutError

```python
class MountUploadTimeoutError(modal.exception.TimeoutError)
```

Raised when a Mount upload times out.

## modal.exception.NotFoundError

```python
class NotFoundError(modal.exception.Error)
```

Raised when a requested resource was not found.

## modal.exception.OutputExpiredError

```python
class OutputExpiredError(modal.exception.TimeoutError)
```

Raised when the Output exceeds expiration and times out.

## modal.exception.PendingDeprecationError

```python
class PendingDeprecationError(UserWarning)
```

Soon to be deprecated feature. Only used intermittently because of multi-repo concerns.

## modal.exception.RemoteError

```python
class RemoteError(modal.exception.Error)
```

Raised when an error occurs on the Modal server.

## modal.exception.RequestSizeError

```python
class RequestSizeError(modal.exception.Error)
```

Raised when an operation produces a gRPC request that is rejected by the server for being too large.

## modal.exception.SandboxTerminatedError

```python
class SandboxTerminatedError(modal.exception.Error)
```

Raised when a Sandbox is terminated for an internal reason.

## modal.exception.SandboxTimeoutError

```python
class SandboxTimeoutError(modal.exception.TimeoutError)
```

Raised when a Sandbox exceeds its execution duration limit and times out.

## modal.exception.SerializationError

```python
class SerializationError(modal.exception.Error)
```

Raised to provide more context when an error is encountered during serialization.

## modal.exception.ServerWarning

```python
class ServerWarning(UserWarning)
```

Warning originating from the Modal server and re-issued in client code.

## modal.exception.TimeoutError

```python
class TimeoutError(modal.exception.Error)
```

Base class for Modal timeouts.

## modal.exception.VersionError

```python
class VersionError(modal.exception.Error)
```

Raised when the current client version of Modal is unsupported.

## modal.exception.VolumeUploadTimeoutError

```python
class VolumeUploadTimeoutError(modal.exception.TimeoutError)
```

Raised when a Volume upload times out.

## modal.exception.simulate_preemption

```python
def simulate_preemption(wait_seconds: int, jitter_seconds: int = 0):
```

Utility for simulating a preemption interrupt after `wait_seconds` seconds.
The first interrupt is the SIGINT signal. After 30 seconds, a second
interrupt will trigger.

This second interrupt simulates SIGKILL, and should not be caught.
Optionally add between zero and `jitter_seconds` seconds of additional waiting before first interrupt.

**Usage:**

```python notest
import time
from modal.exception import simulate_preemption

simulate_preemption(3)

try:
    time.sleep(4)
except KeyboardInterrupt:
    print("got preempted") # Handle interrupt
    raise
```

See https://modal.com/docs/guide/preemption for more details on preemption
handling.

#### exit

# modal.exit

```python
def exit(_warn_parentheses_missing=None) -> Callable[[NullaryMethod], _PartialFunction]:
```

Decorator for methods which should be executed when a container is about to exit.

See the [lifeycle function guide](https://modal.com/docs/guide/lifecycle-functions#exit) for more information.

#### fastapi endpoint

# modal.fastapi_endpoint

```python
def fastapi_endpoint(
    *,
    method: str = "GET",  # REST method for the created endpoint.
    label: Optional[str] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
    custom_domains: Optional[Iterable[str]] = None,  # Custom fully-qualified domain name (FQDN) for the endpoint.
    docs: bool = False,  # Whether to enable interactive documentation for this endpoint at /docs.
    requires_proxy_auth: bool = False,  # Require Modal-Key and Modal-Secret HTTP Headers on requests.
) -> Callable[
    [Union[_PartialFunction[P, ReturnType, ReturnType], Callable[P, ReturnType]]],
    _PartialFunction[P, ReturnType, ReturnType],
]:
```

Convert a function into a basic web endpoint by wrapping it with a FastAPI App.

Modal will internally use [FastAPI](https://fastapi.tiangolo.com/) to expose a
simple, single request handler. If you are defining your own `FastAPI` application
(e.g. if you want to define multiple routes), use `@modal.asgi_app` instead.

The endpoint created with this decorator will automatically have
[CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS) enabled
and can leverage many of FastAPI's features.

For more information on using Modal with popular web frameworks, see our
[guide on web endpoints](https://modal.com/docs/guide/webhooks).

*Added in v0.73.82*: This function replaces the deprecated `@web_endpoint` decorator.

#### file io

# modal.file_io

## modal.file_io.FileIO

```python
class FileIO(typing.Generic)
```

[Alpha] FileIO handle, used in the Sandbox filesystem API.

The API is designed to mimic Python's io.FileIO.

Currently this API is in Alpha and is subject to change. File I/O operations
may be limited in size to 100 MiB, and the throughput of requests is
restricted in the current implementation. For our recommendations on large file transfers
see the Sandbox [filesystem access guide](https://modal.com/docs/guide/sandbox-files).

**Usage**

```python notest
import modal

app = modal.App.lookup("my-app", create_if_missing=True)

sb = modal.Sandbox.create(app=app)
f = sb.open("/tmp/foo.txt", "w")
f.write("hello")
f.close()
```

```python
def __init__(self, client: _Client, task_id: str) -> None:
```

### create

```python
@classmethod
def create(
    cls, path: str, mode: Union["_typeshed.OpenTextMode", "_typeshed.OpenBinaryMode"], client: _Client, task_id: str
) -> "_FileIO":
```

Create a new FileIO handle.
### read

```python
def read(self, n: Optional[int] = None) -> T:
```

Read n bytes from the current position, or the entire remaining file if n is None.
### readline

```python
def readline(self) -> T:
```

Read a single line from the current position.
### readlines

```python
def readlines(self) -> Sequence[T]:
```

Read all lines from the current position.
### write

```python
def write(self, data: Union[bytes, str]) -> None:
```

Write data to the current position.

Writes may not appear until the entire buffer is flushed, which
can be done manually with `flush()` or automatically when the file is
closed.
### flush

```python
def flush(self) -> None:
```

Flush the buffer to disk.
### seek

```python
def seek(self, offset: int, whence: int = 0) -> None:
```

Move to a new position in the file.

`whence` defaults to 0 (absolute file positioning); other values are 1
(relative to the current position) and 2 (relative to the file's end).
### ls

```python
@classmethod
def ls(cls, path: str, client: _Client, task_id: str) -> list[str]:
```

List the contents of the provided directory.
### mkdir

```python
@classmethod
def mkdir(cls, path: str, client: _Client, task_id: str, parents: bool = False) -> None:
```

Create a new directory.
### rm

```python
@classmethod
def rm(cls, path: str, client: _Client, task_id: str, recursive: bool = False) -> None:
```

Remove a file or directory in the Sandbox.
### watch

```python
@classmethod
def watch(
    cls,
    path: str,
    client: _Client,
    task_id: str,
    filter: Optional[list[FileWatchEventType]] = None,
    recursive: bool = False,
    timeout: Optional[int] = None,
) -> Iterator[FileWatchEvent]:
```

### close

```python
def close(self) -> None:
```

Flush the buffer and close the file.
## modal.file_io.FileWatchEvent

```python
class FileWatchEvent(object)
```

FileWatchEvent(paths: list[str], type: modal.file_io.FileWatchEventType)

```python
def __init__(self, paths: list[str], type: modal.file_io.FileWatchEventType) -> None
```

## modal.file_io.FileWatchEventType

```python
class FileWatchEventType(enum.Enum)
```

An enumeration.

The possible values are:

* `Unknown`
* `Access`
* `Create`
* `Modify`
* `Remove`
## modal.file_io.delete_bytes

```python
async def delete_bytes(file: "_FileIO", start: Optional[int] = None, end: Optional[int] = None) -> None:
```

Delete a range of bytes from the file.

`start` and `end` are byte offsets. `start` is inclusive, `end` is exclusive.
If either is None, the start or end of the file is used, respectively.
## modal.file_io.replace_bytes

```python
async def replace_bytes(file: "_FileIO", data: bytes, start: Optional[int] = None, end: Optional[int] = None) -> None:
```

Replace a range of bytes in the file with new data. The length of the data does not
have to be the same as the length of the range being replaced.

`start` and `end` are byte offsets. `start` is inclusive, `end` is exclusive.
If either is None, the start or end of the file is used, respectively.

#### forward

# modal.forward

```python
@contextmanager
def forward(port: int, *, unencrypted: bool = False, client: Optional[_Client] = None) -> Iterator[Tunnel]:
```

Expose a port publicly from inside a running Modal container, with TLS.

If `unencrypted` is set, this also exposes the TCP socket without encryption on a random port
number. This can be used to SSH into a container (see example below). Note that it is on the public Internet, so
make sure you are using a secure protocol over TCP.

**Important:** This is an experimental API which may change in the future.

**Usage:**

```python notest
import modal
from flask import Flask

app = modal.App(image=modal.Image.debian_slim().pip_install("Flask"))
flask_app = Flask(__name__)

@flask_app.route("/")
def hello_world():
    return "Hello, World!"

@app.function()
def run_app():
    # Start a web server inside the container at port 8000. `modal.forward(8000)` lets us
    # expose that port to the world at a random HTTPS URL.
    with modal.forward(8000) as tunnel:
        print("Server listening at", tunnel.url)
        flask_app.run("0.0.0.0", 8000)

    # When the context manager exits, the port is no longer exposed.
```

**Raw TCP usage:**

```python
import socket
import threading

import modal

def run_echo_server(port: int):
    """Run a TCP echo server listening on the given port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", port))
    sock.listen(1)

    while True:
        conn, addr = sock.accept()
        print("Connection from:", addr)

        # Start a new thread to handle the connection
        def handle(conn):
            with conn:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    conn.sendall(data)

        threading.Thread(target=handle, args=(conn,)).start()

app = modal.App()

@app.function()
def tcp_tunnel():
    # This exposes port 8000 to public Internet traffic over TCP.
    with modal.forward(8000, unencrypted=True) as tunnel:
        # You can connect to this TCP socket from outside the container, for example, using `nc`:
        #  nc <HOST> <PORT>
        print("TCP tunnel listening at:", tunnel.tcp_socket)
        run_echo_server(8000)
```

**SSH example:**
This assumes you have a rsa keypair in `~/.ssh/id_rsa{.pub}`, this is a bare-bones example
letting you SSH into a Modal container.

```python
import subprocess
import time

import modal

app = modal.App()
image = (
    modal.Image.debian_slim()
    .apt_install("openssh-server")
    .run_commands("mkdir /run/sshd")
    .add_local_file("~/.ssh/id_rsa.pub", "/root/.ssh/authorized_keys", copy=True)
)

@app.function(image=image, timeout=3600)
def some_function():
    subprocess.Popen(["/usr/sbin/sshd", "-D", "-e"])
    with modal.forward(port=22, unencrypted=True) as tunnel:
        hostname, port = tunnel.tcp_socket
        connection_cmd = f'ssh -p {port} root@{hostname}'
        print(f"ssh into container using: {connection_cmd}")
        time.sleep(3600)  # keep alive for 1 hour or until killed
```

If you intend to use this more generally, a suggestion is to put the subprocess and port
forwarding code in an `@enter` lifecycle method of an @app.cls, to only make a single
ssh server and port for each container (and not one for each input to the function).

#### gpu

# modal.gpu

**GPU configuration shortcodes**

You can pass a wide range of `str` values for the `gpu` parameter of
[`@app.function`](https://modal.com/docs/reference/modal.App#function).

For instance:
- `gpu="H100"` will attach 1 H100 GPU to each container
- `gpu="L40S"` will attach 1 L40S GPU to each container
- `gpu="T4:4"` will attach 4 T4 GPUs to each container

You can see a list of Modal GPU options in the
[GPU docs](https://modal.com/docs/guide/gpu).

**Example**

```python
@app.function(gpu="A100-80GB:4")
def my_gpu_function():
    ... # This will have 4 A100-80GB with each container
```

**Deprecation notes**

An older deprecated way to configure GPU is also still supported,
but will be removed in future versions of Modal. Examples:

- `gpu=modal.gpu.H100()` will attach 1 H100 GPU to each container
- `gpu=modal.gpu.T4(count=4)` will attach 4 T4 GPUs to each container
- `gpu=modal.gpu.A100()` will attach 1 A100-40GB GPUs to each container
- `gpu=modal.gpu.A100(size="80GB")` will attach 1 A100-80GB GPUs to each container

## modal.gpu.A100

```python
class A100(modal.gpu._GPUConfig)
```

[NVIDIA A100 Tensor Core](https://www.nvidia.com/en-us/data-center/a100/) GPU class.

The flagship data center GPU of the Ampere architecture. Available in 40GB and 80GB GPU memory configurations.

```python
def __init__(
    self,
    *,
    count: int = 1,  # Number of GPUs per container. Defaults to 1.
    size: Union[str, None] = None,  # Select GB configuration of GPU device: "40GB" or "80GB". Defaults to "40GB".
):
```

## modal.gpu.A10G

```python
class A10G(modal.gpu._GPUConfig)
```

[NVIDIA A10G Tensor Core](https://www.nvidia.com/en-us/data-center/products/a10-gpu/) GPU class.

A mid-tier data center GPU based on the Ampere architecture, providing 24 GB of memory.
A10G GPUs deliver up to 3.3x better ML training performance, 3x better ML inference performance,
and 3x better graphics performance, in comparison to NVIDIA T4 GPUs.

```python
def __init__(
    self,
    *,
    # Number of GPUs per container. Defaults to 1.
    # Useful if you have very large models that don't fit on a single GPU.
    count: int = 1,
):
```

## modal.gpu.Any

```python
class Any(modal.gpu._GPUConfig)
```

Selects any one of the GPU classes available within Modal, according to availability.

```python
def __init__(self, *, count: int = 1):
```

## modal.gpu.H100

```python
class H100(modal.gpu._GPUConfig)
```

[NVIDIA H100 Tensor Core](https://www.nvidia.com/en-us/data-center/h100/) GPU class.

The flagship data center GPU of the Hopper architecture.
Enhanced support for FP8 precision and a Transformer Engine that provides up to 4X faster training
over the prior generation for GPT-3 (175B) models.

```python
def __init__(
    self,
    *,
    # Number of GPUs per container. Defaults to 1.
    # Useful if you have very large models that don't fit on a single GPU.
    count: int = 1,
):
```

## modal.gpu.L4

```python
class L4(modal.gpu._GPUConfig)
```

[NVIDIA L4 Tensor Core](https://www.nvidia.com/en-us/data-center/l4/) GPU class.

A mid-tier data center GPU based on the Ada Lovelace architecture, providing 24GB of GPU memory.
Includes RTX (ray tracing) support.

```python
def __init__(
    self,
    count: int = 1,  # Number of GPUs per container. Defaults to 1.
):
```

## modal.gpu.L40S

```python
class L40S(modal.gpu._GPUConfig)
```

[NVIDIA L40S](https://www.nvidia.com/en-us/data-center/l40s/) GPU class.

The L40S is a data center GPU for the Ada Lovelace architecture. It has 48 GB of on-chip
GDDR6 RAM and enhanced support for FP8 precision.

```python
def __init__(
    self,
    *,
    # Number of GPUs per container. Defaults to 1.
    # Useful if you have very large models that don't fit on a single GPU.
    count: int = 1,
):
```

## modal.gpu.T4

```python
class T4(modal.gpu._GPUConfig)
```

[NVIDIA T4 Tensor Core](https://www.nvidia.com/en-us/data-center/tesla-t4/) GPU class.

A low-cost data center GPU based on the Turing architecture, providing 16GB of GPU memory.

```python
def __init__(
    self,
    count: int = 1,  # Number of GPUs per container. Defaults to 1.
):
```

## modal.gpu.parse_gpu_config

```python
def parse_gpu_config(value: GPU_T) -> api_pb2.GPUConfig:
```

#### interact

# modal.interact

```python
def interact() -> None:
```

Enable interactivity with user input inside a Modal container.

See the [interactivity guide](https://modal.com/docs/guide/developing-debugging#interactivity)
for more information on how to use this function.

#### io streams

# modal.io_streams

## modal.io_streams.StreamReader

```python
class StreamReader(typing.Generic)
```

Retrieve logs from a stream (`stdout` or `stderr`).

As an asynchronous iterable, the object supports the `for` and `async for`
statements. Just loop over the object to read in chunks.

**Usage**

```python fixture:running_app
from modal import Sandbox

sandbox = Sandbox.create(
    "bash",
    "-c",
    "for i in $(seq 1 10); do echo foo; sleep 0.1; done",
    app=running_app,
)
for message in sandbox.stdout:
    print(f"Message: {message}")
```

### file_descriptor

```python
@property
def file_descriptor(self) -> int:
```

Possible values are `1` for stdout and `2` for stderr.
### read

```python
def read(self) -> T:
```

Fetch the entire contents of the stream until EOF.

**Usage**

```python fixture:running_app
from modal import Sandbox

sandbox = Sandbox.create("echo", "hello", app=running_app)
sandbox.wait()

print(sandbox.stdout.read())
```
## modal.io_streams.StreamWriter

```python
class StreamWriter(object)
```

Provides an interface to buffer and write logs to a sandbox or container process stream (`stdin`).

### write

```python
def write(self, data: Union[bytes, bytearray, memoryview, str]) -> None:
```

Write data to the stream but does not send it immediately.

This is non-blocking and queues the data to an internal buffer. Must be
used along with the `drain()` method, which flushes the buffer.

**Usage**

```python fixture:running_app
from modal import Sandbox

sandbox = Sandbox.create(
    "bash",
    "-c",
    "while read line; do echo $line; done",
    app=running_app,
)
sandbox.stdin.write(b"foo\n")
sandbox.stdin.write(b"bar\n")
sandbox.stdin.write_eof()

sandbox.stdin.drain()
sandbox.wait()
```
### write_eof

```python
def write_eof(self) -> None:
```

Close the write end of the stream after the buffered data is drained.

If the process was blocked on input, it will become unblocked after
`write_eof()`. This method needs to be used along with the `drain()`
method, which flushes the EOF to the process.
### drain

```python
def drain(self) -> None:
```

Flush the write buffer and send data to the running process.

This is a flow control method that blocks until data is sent. It returns
when it is appropriate to continue writing data to the stream.

**Usage**

```python notest
writer.write(data)
writer.drain()
```

Async usage:
```python notest
writer.write(data)  # not a blocking operation
await writer.drain.aio()
```

#### is local

# modal.is_local

```python
def is_local() -> bool:
```

Returns if we are currently on the machine launching/deploying a Modal app

Returns `True` when executed locally on the user's machine.
Returns `False` when executed from a Modal container in the cloud.

#### method

# modal.method

```python
def method(
    *,
    # Set this to True if it's a non-generator function returning
    # a [sync/async] generator object
    is_generator: Optional[bool] = None,
) -> _MethodDecoratorType:
```

Decorator for methods that should be transformed into a Modal Function registered against this class's App.

**Usage:**

```python
@app.cls(cpu=8)
class MyCls:

    @modal.method()
    def f(self):
        ...
```

#### parameter

# modal.parameter

```python
def parameter(*, default: Any = _no_default, init: bool = True) -> Any:
```

Used to specify options for modal.cls parameters, similar to dataclass.field for dataclasses
```
class A:
    a: str = modal.parameter()

```

If `init=False` is specified, the field is not considered a parameter for the
Modal class and not used in the synthesized constructor. This can be used to
optionally annotate the type of a field that's used internally, for example values
being set by @enter lifecycle methods, without breaking type checkers, but it has
no runtime effect on the class.

#### web endpoint

# modal.web_endpoint

```python
def web_endpoint(
    *,
    method: str = "GET",  # REST method for the created endpoint.
    label: Optional[str] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
    docs: bool = False,  # Whether to enable interactive documentation for this endpoint at /docs.
    custom_domains: Optional[
        Iterable[str]
    ] = None,  # Create an endpoint using a custom domain fully-qualified domain name (FQDN).
    requires_proxy_auth: bool = False,  # Require Modal-Key and Modal-Secret HTTP Headers on requests.
) -> Callable[
    [Union[_PartialFunction[P, ReturnType, ReturnType], Callable[P, ReturnType]]],
    _PartialFunction[P, ReturnType, ReturnType],
]:
```

Register a basic web endpoint with this application.

DEPRECATED: This decorator has been renamed to `@modal.fastapi_endpoint`.

This is the simple way to create a web endpoint on Modal. The function
behaves as a [FastAPI](https://fastapi.tiangolo.com/) handler and should
return a response object to the caller.

Endpoints created with `@modal.web_endpoint` are meant to be simple, single
request handlers and automatically have
[CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS) enabled.
For more flexibility, use `@modal.asgi_app`.

To learn how to use Modal with popular web frameworks, see the
[guide on web endpoints](https://modal.com/docs/guide/webhooks).

#### web server

# modal.web_server

```python
def web_server(
    port: int,
    *,
    startup_timeout: float = 5.0,  # Maximum number of seconds to wait for the web server to start.
    label: Optional[str] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
    custom_domains: Optional[Iterable[str]] = None,  # Deploy this endpoint on a custom domain.
    requires_proxy_auth: bool = False,  # Require Modal-Key and Modal-Secret HTTP Headers on requests.
) -> Callable[[Union[_PartialFunction, NullaryFuncOrMethod]], _PartialFunction]:
```

Decorator that registers an HTTP web server inside the container.

This is similar to `@asgi_app` and `@wsgi_app`, but it allows you to expose a full HTTP server
listening on a container port. This is useful for servers written in other languages like Rust,
as well as integrating with non-ASGI frameworks like aiohttp and Tornado.

**Usage:**

```python
import subprocess

@app.function()
@modal.web_server(8000)
def my_file_server():
    subprocess.Popen("python -m http.server -d / 8000", shell=True)
```

The above example starts a simple file server, displaying the contents of the root directory.
Here, requests to the web endpoint will go to external port 8000 on the container. The
`http.server` module is included with Python, but you could run anything here.

Internally, the web server is transparently converted into a web endpoint by Modal, so it has
the same serverless autoscaling behavior as other web endpoints.

For more info, see the [guide on web endpoints](https://modal.com/docs/guide/webhooks).

#### wsgi app

# modal.wsgi_app

```python
def wsgi_app(
    *,
    label: Optional[str] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
    custom_domains: Optional[Iterable[str]] = None,  # Deploy this endpoint on a custom domain.
    requires_proxy_auth: bool = False,  # Require Modal-Key and Modal-Secret HTTP Headers on requests.
) -> Callable[[Union[_PartialFunction, NullaryFuncOrMethod]], _PartialFunction]:
```

Decorator for registering a WSGI app with a Modal function.

Web Server Gateway Interface (WSGI) is a standard for synchronous Python web apps.
It has been [succeeded by the ASGI interface](https://asgi.readthedocs.io/en/latest/introduction.html#wsgi-compatibility)
which is compatible with ASGI and supports additional functionality such as web sockets.
Modal supports ASGI via [`asgi_app`](https://modal.com/docs/reference/modal.asgi_app).

**Usage:**

```python
from typing import Callable

@app.function()
@modal.wsgi_app()
def create_wsgi() -> Callable:
    ...
```

To learn how to use this decorator with popular web frameworks, see the
[guide on web endpoints](https://modal.com/docs/guide/webhooks).

### CLI Reference

### modal app

# `modal app`

Manage deployed and running apps.

**Usage**:

```shell
modal app [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `list`: List Modal apps that are currently deployed/running or recently stopped.
* `logs`: Show App logs, streaming while active.
* `rollback`: Redeploy a previous version of an App.
* `stop`: Stop an app.
* `history`: Show App deployment history, for a currently deployed app

## `modal app list`

List Modal apps that are currently deployed/running or recently stopped.

**Usage**:

```shell
modal app list [OPTIONS]
```

**Options**:

* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--json / --no-json`: \[default: no-json]
* `--help`: Show this message and exit.

## `modal app logs`

Show App logs, streaming while active.

**Examples:**

Get the logs based on an app ID:

```
modal app logs ap-123456
```

Get the logs for a currently deployed App based on its name:

```
modal app logs my-app
```

**Usage**:

```shell
modal app logs [OPTIONS] [APP_IDENTIFIER]
```

**Arguments**:

* `[APP_IDENTIFIER]`: App name or ID

**Options**:

* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--timestamps`: Show timestamps for each log line
* `--help`: Show this message and exit.

## `modal app rollback`

Redeploy a previous version of an App.

Note that the App must currently be in a "deployed" state.
Rollbacks will appear as a new deployment in the App history, although
the App state will be reset to the state at the time of the previous deployment.

**Examples:**

Rollback an App to its previous version:

```
modal app rollback my-app
```

Rollback an App to a specific version:

```
modal app rollback my-app v3
```

Rollback an App using its App ID instead of its name:

```
modal app rollback ap-abcdefghABCDEFGH123456
```

**Usage**:

```shell
modal app rollback [OPTIONS] [APP_IDENTIFIER] [VERSION]
```

**Arguments**:

* `[APP_IDENTIFIER]`: App name or ID
* `[VERSION]`: Target version for rollback.

**Options**:

* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

## `modal app stop`

Stop an app.

**Usage**:

```shell
modal app stop [OPTIONS] [APP_IDENTIFIER]
```

**Arguments**:

* `[APP_IDENTIFIER]`: App name or ID

**Options**:

* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

## `modal app history`

Show App deployment history, for a currently deployed app

**Examples:**

Get the history based on an app ID:

```
modal app history ap-123456
```

Get the history for a currently deployed App based on its name:

```
modal app history my-app
```

**Usage**:

```shell
modal app history [OPTIONS] [APP_IDENTIFIER]
```

**Arguments**:

* `[APP_IDENTIFIER]`: App name or ID

**Options**:

* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--json / --no-json`: \[default: no-json]
* `--help`: Show this message and exit.

### modal config

# `modal config`

Manage client configuration for the current profile.

Refer to https://modal.com/docs/reference/modal.config for a full explanation
of what these options mean, and how to set them.

**Usage**:

```shell
modal config [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `show`: Show current configuration values (debugging command).
* `set-environment`: Set the default Modal environment for the active profile

## `modal config show`

Show current configuration values (debugging command).

**Usage**:

```shell
modal config show [OPTIONS]
```

**Options**:

* `--redact / --no-redact`: Redact the `token_secret` value.  \[default: redact]
* `--help`: Show this message and exit.

## `modal config set-environment`

Set the default Modal environment for the active profile

The default environment of a profile is used when no --env flag is passed to `modal run`, `modal deploy` etc.

If no default environment is set, and there exists multiple environments in a workspace, an error will be raised
when running a command that requires an environment.

**Usage**:

```shell
modal config set-environment [OPTIONS] ENVIRONMENT_NAME
```

**Arguments**:

* `ENVIRONMENT_NAME`: \[required]

**Options**:

* `--help`: Show this message and exit.

### modal container

# `modal container`

Manage and connect to running containers.

**Usage**:

```shell
modal container [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `list`: List all containers that are currently running.
* `logs`: Show logs for a specific container, streaming while active.
* `exec`: Execute a command in a container.
* `stop`: Stop a currently-running container and reassign its in-progress inputs.

## `modal container list`

List all containers that are currently running.

**Usage**:

```shell
modal container list [OPTIONS]
```

**Options**:

* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--json / --no-json`: \[default: no-json]
* `--help`: Show this message and exit.

## `modal container logs`

Show logs for a specific container, streaming while active.

**Usage**:

```shell
modal container logs [OPTIONS] CONTAINER_ID
```

**Arguments**:

* `CONTAINER_ID`: Container ID  \[required]

**Options**:

* `--help`: Show this message and exit.

## `modal container exec`

Execute a command in a container.

**Usage**:

```shell
modal container exec [OPTIONS] CONTAINER_ID COMMAND...
```

**Arguments**:

* `CONTAINER_ID`: Container ID  \[required]
* `COMMAND...`: A command to run inside the container.

To pass command-line flags or options, add `--` before the start of your commands. For example: `modal container exec <id> -- /bin/bash -c 'echo hi'`  \[required]

**Options**:

* `--pty / --no-pty`: Run the command using a PTY.
* `--help`: Show this message and exit.

## `modal container stop`

Stop a currently-running container and reassign its in-progress inputs.

This will send the container a SIGINT signal that Modal will handle.

**Usage**:

```shell
modal container stop [OPTIONS] CONTAINER_ID
```

**Arguments**:

* `CONTAINER_ID`: Container ID  \[required]

**Options**:

* `--help`: Show this message and exit.

### modal deploy

# `modal deploy`

Deploy a Modal application.

**Usage:**
modal deploy my_script.py
modal deploy -m my_package.my_mod

**Usage**:

```shell
modal deploy [OPTIONS] APP_REF
```

**Arguments**:

* `APP_REF`: Path to a Python file with an app to deploy  \[required]

**Options**:

* `--name TEXT`: Name of the deployment.
* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--stream-logs / --no-stream-logs`: Stream logs from the app upon deployment.  \[default: no-stream-logs]
* `--tag TEXT`: Tag the deployment with a version.
* `-m`: Interpret argument as a Python module path instead of a file/script path
* `--help`: Show this message and exit.

### modal dict

# `modal dict`

Manage `modal.Dict` objects and inspect their contents.

**Usage**:

```shell
modal dict [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `create`: Create a named Dict object.
* `list`: List all named Dicts.
* `clear`: Clear the contents of a named Dict by deleting all of its data.
* `delete`: Delete a named Dict and all of its data.
* `get`: Print the value for a specific key.
* `items`: Print the contents of a Dict.

## `modal dict create`

Create a named Dict object.

Note: This is a no-op when the Dict already exists.

**Usage**:

```shell
modal dict create [OPTIONS] NAME
```

**Arguments**:

* `NAME`: \[required]

**Options**:

* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

## `modal dict list`

List all named Dicts.

**Usage**:

```shell
modal dict list [OPTIONS]
```

**Options**:

* `--json / --no-json`: \[default: no-json]
* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

## `modal dict clear`

Clear the contents of a named Dict by deleting all of its data.

**Usage**:

```shell
modal dict clear [OPTIONS] NAME
```

**Arguments**:

* `NAME`: \[required]

**Options**:

* `-y, --yes`: Run without pausing for confirmation.
* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

## `modal dict delete`

Delete a named Dict and all of its data.

**Usage**:

```shell
modal dict delete [OPTIONS] NAME
```

**Arguments**:

* `NAME`: \[required]

**Options**:

* `--allow-missing`: Don't error if the Dict doesn't exist.
* `-y, --yes`: Run without pausing for confirmation.
* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

## `modal dict get`

Print the value for a specific key.

Note: When using the CLI, keys are always interpreted as having a string type.

**Usage**:

```shell
modal dict get [OPTIONS] NAME KEY
```

**Arguments**:

* `NAME`: \[required]
* `KEY`: \[required]

**Options**:

* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

## `modal dict items`

Print the contents of a Dict.

Note: By default, this command truncates the contents. Use the `N` argument to control the
amount of data shown or the `--all` option to retrieve the entire Dict, which may be slow.

**Usage**:

```shell
modal dict items [OPTIONS] NAME [N]
```

**Arguments**:

* `NAME`: \[required]
* `[N]`: Limit the number of entries shown  \[default: 20]

**Options**:

* `-a, --all`: Ignore N and print all entries in the Dict (may be slow)
* `-r, --repr`: Display items using `repr()` to see more details
* `--json / --no-json`: \[default: no-json]
* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

### modal environment

# `modal environment`

Create and interact with Environments

Environments are sub-divisons of workspaces, allowing you to deploy the same app
in different namespaces. Each environment has their own set of Secrets and any
lookups performed from an app in an environment will by default look for entities
in the same environment.

Typical use cases for environments include having one for development and one for
production, to prevent overwriting production apps when developing new features
while still being able to deploy changes to a live environment.

**Usage**:

```shell
modal environment [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `list`: List all environments in the current workspace
* `create`: Create a new environment in the current workspace
* `delete`: Delete an environment in the current workspace
* `update`: Update the name or web suffix of an environment

## `modal environment list`

List all environments in the current workspace

**Usage**:

```shell
modal environment list [OPTIONS]
```

**Options**:

* `--json / --no-json`: \[default: no-json]
* `--help`: Show this message and exit.

## `modal environment create`

Create a new environment in the current workspace

**Usage**:

```shell
modal environment create [OPTIONS] NAME
```

**Arguments**:

* `NAME`: Name of the new environment. Must be unique. Case sensitive  \[required]

**Options**:

* `--help`: Show this message and exit.

## `modal environment delete`

Delete an environment in the current workspace

Deletes all apps in the selected environment and deletes the environment irrevocably.

**Usage**:

```shell
modal environment delete [OPTIONS] NAME
```

**Arguments**:

* `NAME`: Name of the environment to be deleted. Case sensitive  \[required]

**Options**:

* `--confirm / --no-confirm`: Set this flag to delete without prompting for confirmation  \[default: no-confirm]
* `--help`: Show this message and exit.

## `modal environment update`

Update the name or web suffix of an environment

**Usage**:

```shell
modal environment update [OPTIONS] CURRENT_NAME
```

**Arguments**:

* `CURRENT_NAME`: \[required]

**Options**:

* `--set-name TEXT`: New name of the environment
* `--set-web-suffix TEXT`: New web suffix of environment (empty string is no suffix)
* `--help`: Show this message and exit.

### modal launch

# `modal launch`

Open a serverless app instance on Modal.
>⚠️  `modal launch` is **experimental** and may change in the future.

**Usage**:

```shell
modal launch [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `jupyter`: Start Jupyter Lab on Modal.
* `vscode`: Start Visual Studio Code on Modal.

## `modal launch jupyter`

Start Jupyter Lab on Modal.

**Usage**:

```shell
modal launch jupyter [OPTIONS]
```

**Options**:

* `--cpu INTEGER`: \[default: 8]
* `--memory INTEGER`: \[default: 32768]
* `--gpu TEXT`
* `--timeout INTEGER`: \[default: 3600]
* `--image TEXT`: \[default: ubuntu:22.04]
* `--add-python TEXT`: \[default: 3.11]
* `--mount TEXT`
* `--volume TEXT`
* `--detach / --no-detach`: \[default: no-detach]
* `--help`: Show this message and exit.

## `modal launch vscode`

Start Visual Studio Code on Modal.

**Usage**:

```shell
modal launch vscode [OPTIONS]
```

**Options**:

* `--cpu INTEGER`: \[default: 8]
* `--memory INTEGER`: \[default: 32768]
* `--gpu TEXT`
* `--image TEXT`: \[default: debian:12]
* `--timeout INTEGER`: \[default: 3600]
* `--mount TEXT`
* `--volume TEXT`
* `--detach / --no-detach`: \[default: no-detach]
* `--help`: Show this message and exit.

### modal nfs

# `modal nfs`

Read and edit `modal.NetworkFileSystem` file systems.

**Usage**:

```shell
modal nfs [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `list`: List the names of all network file systems.
* `create`: Create a named network file system.
* `ls`: List files and directories in a network file system.
* `put`: Upload a file or directory to a network file system.
* `get`: Download a file from a network file system.
* `rm`: Delete a file or directory from a network file system.
* `delete`: Delete a named, persistent modal.NetworkFileSystem.

## `modal nfs list`

List the names of all network file systems.

**Usage**:

```shell
modal nfs list [OPTIONS]
```

**Options**:

* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--json / --no-json`: \[default: no-json]
* `--help`: Show this message and exit.

## `modal nfs create`

Create a named network file system.

**Usage**:

```shell
modal nfs create [OPTIONS] NAME
```

**Arguments**:

* `NAME`: \[required]

**Options**:

* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

## `modal nfs ls`

List files and directories in a network file system.

**Usage**:

```shell
modal nfs ls [OPTIONS] VOLUME_NAME [PATH]
```

**Arguments**:

* `VOLUME_NAME`: \[required]
* `[PATH]`: \[default: /]

**Options**:

* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

## `modal nfs put`

Upload a file or directory to a network file system.

Remote parent directories will be created as needed.

Ending the REMOTE_PATH with a forward slash (/), it's assumed to be a directory and the file
will be uploaded with its current name under that directory.

**Usage**:

```shell
modal nfs put [OPTIONS] VOLUME_NAME LOCAL_PATH [REMOTE_PATH]
```

**Arguments**:

* `VOLUME_NAME`: \[required]
* `LOCAL_PATH`: \[required]
* `[REMOTE_PATH]`: \[default: /]

**Options**:

* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

## `modal nfs get`

Download a file from a network file system.

Specifying a glob pattern (using any `*` or `**` patterns) as the `remote_path` will download
all matching files, preserving their directory structure.

For example, to download an entire network file system into `dump_volume`:

```
modal nfs get <volume-name> "**" dump_volume
```

Use "-" as LOCAL_DESTINATION to write file contents to standard output.

**Usage**:

```shell
modal nfs get [OPTIONS] VOLUME_NAME REMOTE_PATH [LOCAL_DESTINATION]
```

**Arguments**:

* `VOLUME_NAME`: \[required]
* `REMOTE_PATH`: \[required]
* `[LOCAL_DESTINATION]`: \[default: .]

**Options**:

* `--force / --no-force`: \[default: no-force]
* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

## `modal nfs rm`

Delete a file or directory from a network file system.

**Usage**:

```shell
modal nfs rm [OPTIONS] VOLUME_NAME REMOTE_PATH
```

**Arguments**:

* `VOLUME_NAME`: \[required]
* `REMOTE_PATH`: \[required]

**Options**:

* `-r, --recursive`: Delete directory recursively
* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

## `modal nfs delete`

Delete a named, persistent modal.NetworkFileSystem.

**Usage**:

```shell
modal nfs delete [OPTIONS] NFS_NAME
```

**Arguments**:

* `NFS_NAME`: Name of the modal.NetworkFileSystem to be deleted. Case sensitive  \[required]

**Options**:

* `-y, --yes`: Run without pausing for confirmation.
* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

### modal profile

# `modal profile`

Switch between Modal profiles.

**Usage**:

```shell
modal profile [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `activate`: Change the active Modal profile.
* `current`: Print the currently active Modal profile.
* `list`: Show all Modal profiles and highlight the active one.

## `modal profile activate`

Change the active Modal profile.

**Usage**:

```shell
modal profile activate [OPTIONS] PROFILE
```

**Arguments**:

* `PROFILE`: Modal profile to activate.  \[required]

**Options**:

* `--help`: Show this message and exit.

## `modal profile current`

Print the currently active Modal profile.

**Usage**:

```shell
modal profile current [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `modal profile list`

Show all Modal profiles and highlight the active one.

**Usage**:

```shell
modal profile list [OPTIONS]
```

**Options**:

* `--json / --no-json`: \[default: no-json]
* `--help`: Show this message and exit.

### modal queue

# `modal queue`

Manage `modal.Queue` objects and inspect their contents.

**Usage**:

```shell
modal queue [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `create`: Create a named Queue.
* `delete`: Delete a named Queue and all of its data.
* `list`: List all named Queues.
* `clear`: Clear the contents of a queue by removing all of its data.
* `peek`: Print the next N items in the queue or queue partition (without removal).
* `len`: Print the length of a queue partition or the total length of all partitions.

## `modal queue create`

Create a named Queue.

Note: This is a no-op when the Queue already exists.

**Usage**:

```shell
modal queue create [OPTIONS] NAME
```

**Arguments**:

* `NAME`: \[required]

**Options**:

* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

## `modal queue delete`

Delete a named Queue and all of its data.

**Usage**:

```shell
modal queue delete [OPTIONS] NAME
```

**Arguments**:

* `NAME`: \[required]

**Options**:

* `--allow-missing`: Don't error if the Queue doesn't exist.
* `-y, --yes`: Run without pausing for confirmation.
* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

## `modal queue list`

List all named Queues.

**Usage**:

```shell
modal queue list [OPTIONS]
```

**Options**:

* `--json / --no-json`: \[default: no-json]
* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

## `modal queue clear`

Clear the contents of a queue by removing all of its data.

**Usage**:

```shell
modal queue clear [OPTIONS] NAME
```

**Arguments**:

* `NAME`: \[required]

**Options**:

* `-p, --partition TEXT`: Name of the partition to use, otherwise use the default (anonymous) partition.
* `-a, --all`: Clear the contents of all partitions.
* `-y, --yes`: Run without pausing for confirmation.
* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

## `modal queue peek`

Print the next N items in the queue or queue partition (without removal).

**Usage**:

```shell
modal queue peek [OPTIONS] NAME [N]
```

**Arguments**:

* `NAME`: \[required]
* `[N]`: \[default: 1]

**Options**:

* `-p, --partition TEXT`: Name of the partition to use, otherwise use the default (anonymous) partition.
* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

## `modal queue len`

Print the length of a queue partition or the total length of all partitions.

**Usage**:

```shell
modal queue len [OPTIONS] NAME
```

**Arguments**:

* `NAME`: \[required]

**Options**:

* `-p, --partition TEXT`: Name of the partition to use, otherwise use the default (anonymous) partition.
* `-t, --total`: Compute the sum of the queue lengths across all partitions
* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

### modal run

# `modal run`

Run a Modal function or local entrypoint.

`FUNC_REF` should be of the format `{file or module}::{function name}`.
Alternatively, you can refer to the function via the app:

`{file or module}::{app variable name}.{function name}`

**Examples:**

To run the hello_world function (or local entrypoint) in my_app.py:

```
modal run my_app.py::hello_world
```

If your module only has a single app and your app has a
single local entrypoint (or single function), you can omit the app and
function parts:

```
modal run my_app.py
```

Instead of pointing to a file, you can also use the Python module path, which
by default will ensure that your remote functions will use the same module
names as they do locally.

```
modal run -m my_project.my_app
```

**Usage**:

```shell
modal run [OPTIONS] FUNC_REF
```

**Options**:

* `-w, --write-result TEXT`: Write return value (which must be str or bytes) to this local path.
* `-q, --quiet`: Don't show Modal progress indicators.
* `-d, --detach`: Don't stop the app if the local process dies or disconnects.
* `-i, --interactive`: Run the app in interactive mode.
* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `-m`: Interpret argument as a Python module path instead of a file/script path
* `--help`: Show this message and exit.

### modal secret

# `modal secret`

Manage secrets.

**Usage**:

```shell
modal secret [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `list`: List your published secrets.
* `create`: Create a new secret.
* `delete`: Delete a named Secret.

## `modal secret list`

List your published secrets.

**Usage**:

```shell
modal secret list [OPTIONS]
```

**Options**:

* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--json / --no-json`: \[default: no-json]
* `--help`: Show this message and exit.

## `modal secret create`

Create a new secret.

**Usage**:

```shell
modal secret create [OPTIONS] SECRET_NAME [KEYVALUES]...
```

**Arguments**:

* `SECRET_NAME`: \[required]
* `[KEYVALUES]...`: Space-separated KEY=VALUE items.

**Options**:

* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--from-dotenv PATH`: Path to a .env file to load secrets from.
* `--from-json PATH`: Path to a JSON file to load secrets from.
* `--force`: Overwrite the secret if it already exists.
* `--help`: Show this message and exit.

## `modal secret delete`

Delete a named Secret.

**Usage**:

```shell
modal secret delete [OPTIONS] NAME
```

**Arguments**:

* `NAME`: Name of the modal.Secret to be deleted. Case sensitive  \[required]

**Options**:

* `--allow-missing`: Don't error if the Secret doesn't exist.
* `-y, --yes`: Run without pausing for confirmation.
* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

### modal serve

# `modal serve`

Run a web endpoint(s) associated with a Modal app and hot-reload code.

**Examples:**

```
modal serve hello_world.py
```

**Usage**:

```shell
modal serve [OPTIONS] APP_REF
```

**Arguments**:

* `APP_REF`: Path to a Python file with an app.  \[required]

**Options**:

* `--timeout FLOAT`
* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `-m`: Interpret argument as a Python module path instead of a file/script path
* `--help`: Show this message and exit.

### modal setup

# `modal setup`

Bootstrap Modal's configuration.

**Usage**:

```shell
modal setup [OPTIONS]
```

**Options**:

* `--profile TEXT`
* `--help`: Show this message and exit.

### modal shell

# `modal shell`

Run a command or interactive shell inside a Modal container.

**Examples:**

Start an interactive shell inside the default Debian-based image:

```
modal shell
```

Start an interactive shell with the spec for `my_function` in your App
(uses the same image, volumes, mounts, etc.):

```
modal shell hello_world.py::my_function
```

Or, if you're using a [modal.Cls](https://modal.com/docs/reference/modal.Cls)
you can refer to a `@modal.method` directly:

```
modal shell hello_world.py::MyClass.my_method
```

Start a `python` shell:

```
modal shell hello_world.py --cmd=python
```

Run a command with your function's spec and pipe the output to a file:

```
modal shell hello_world.py -c 'uv pip list' > env.txt
```

**Usage**:

```shell
modal shell [OPTIONS] REF
```

**Arguments**:

* `REF`: ID of running container, or path to a Python file containing a Modal App. Can also include a function specifier, like `module.py::func`, if the file defines multiple functions.

**Options**:

* `-c, --cmd TEXT`: Command to run inside the Modal image.  \[default: /bin/bash]
* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--image TEXT`: Container image tag for inside the shell (if not using REF).
* `--add-python TEXT`: Add Python to the image (if not using REF).
* `--volume TEXT`: Name of a `modal.Volume` to mount inside the shell at `/mnt/{name}` (if not using REF). Can be used multiple times.
* `--secret TEXT`: Name of a `modal.Secret` to mount inside the shell (if not using REF). Can be used multiple times.
* `--cpu INTEGER`: Number of CPUs to allocate to the shell (if not using REF).
* `--memory INTEGER`: Memory to allocate for the shell, in MiB (if not using REF).
* `--gpu TEXT`: GPUs to request for the shell, if any. Examples are `any`, `a10g`, `a100:4` (if not using REF).
* `--cloud TEXT`: Cloud provider to run the shell on. Possible values are `aws`, `gcp`, `oci`, `auto` (if not using REF).
* `--region TEXT`: Region(s) to run the container on. Can be a single region or a comma-separated list to choose from (if not using REF).
* `--pty / --no-pty`: Run the command using a PTY.
* `-m`: Interpret argument as a Python module path instead of a file/script path
* `--help`: Show this message and exit.

### modal token

# `modal token`

Manage tokens.

**Usage**:

```shell
modal token [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `set`: Set account credentials for connecting to Modal.
* `new`: Create a new token by using an authenticated web session.

## `modal token set`

Set account credentials for connecting to Modal.

If the credentials are not provided on the command line, you will be prompted to enter them.

**Usage**:

```shell
modal token set [OPTIONS]
```

**Options**:

* `--token-id TEXT`: Account token ID.
* `--token-secret TEXT`: Account token secret.
* `--profile TEXT`: Modal profile to set credentials for. If unspecified (and MODAL_PROFILE environment variable is not set), uses the workspace name associated with the credentials.
* `--activate / --no-activate`: Activate the profile containing this token after creation.  \[default: activate]
* `--verify / --no-verify`: Make a test request to verify the new credentials.  \[default: verify]
* `--help`: Show this message and exit.

## `modal token new`

Create a new token by using an authenticated web session.

**Usage**:

```shell
modal token new [OPTIONS]
```

**Options**:

* `--profile TEXT`: Modal profile to set credentials for. If unspecified (and MODAL_PROFILE environment variable is not set), uses the workspace name associated with the credentials.
* `--activate / --no-activate`: Activate the profile containing this token after creation.  \[default: activate]
* `--verify / --no-verify`: Make a test request to verify the new credentials.  \[default: verify]
* `--source TEXT`
* `--help`: Show this message and exit.

### modal volume

# `modal volume`

Read and edit `modal.Volume` volumes.

Note: users of `modal.NetworkFileSystem` should use the `modal nfs` command instead.

**Usage**:

```shell
modal volume [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `create`: Create a named, persistent modal.Volume.
* `get`: Download files from a modal.Volume object.
* `list`: List the details of all modal.Volume volumes in an Environment.
* `ls`: List files and directories in a modal.Volume volume.
* `put`: Upload a file or directory to a modal.Volume.
* `rm`: Delete a file or directory from a modal.Volume.
* `cp`: Copy within a modal.Volume.
* `delete`: Delete a named Volume and all of its data.
* `rename`: Rename a modal.Volume.

## `modal volume create`

Create a named, persistent modal.Volume.

**Usage**:

```shell
modal volume create [OPTIONS] NAME
```

**Arguments**:

* `NAME`: \[required]

**Options**:

* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--version INTEGER`: VolumeFS version. (Experimental)
* `--help`: Show this message and exit.

## `modal volume get`

Download files from a modal.Volume object.

If a folder is passed for REMOTE_PATH, the contents of the folder will be downloaded
recursively, including all subdirectories.

**Example**

```
modal volume get <volume_name> logs/april-12-1.txt
modal volume get <volume_name> / volume_data_dump
```

Use "-" as LOCAL_DESTINATION to write file contents to standard output.

**Usage**:

```shell
modal volume get [OPTIONS] VOLUME_NAME REMOTE_PATH [LOCAL_DESTINATION]
```

**Arguments**:

* `VOLUME_NAME`: \[required]
* `REMOTE_PATH`: \[required]
* `[LOCAL_DESTINATION]`: \[default: .]

**Options**:

* `--force / --no-force`: \[default: no-force]
* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

## `modal volume list`

List the details of all modal.Volume volumes in an Environment.

**Usage**:

```shell
modal volume list [OPTIONS]
```

**Options**:

* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--json / --no-json`: \[default: no-json]
* `--help`: Show this message and exit.

## `modal volume ls`

List files and directories in a modal.Volume volume.

**Usage**:

```shell
modal volume ls [OPTIONS] VOLUME_NAME [PATH]
```

**Arguments**:

* `VOLUME_NAME`: \[required]
* `[PATH]`: \[default: /]

**Options**:

* `--json / --no-json`: \[default: no-json]
* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

## `modal volume put`

Upload a file or directory to a modal.Volume.

Remote parent directories will be created as needed.

Ending the REMOTE_PATH with a forward slash (/), it's assumed to be a directory
and the file will be uploaded with its current name under that directory.

**Usage**:

```shell
modal volume put [OPTIONS] VOLUME_NAME LOCAL_PATH [REMOTE_PATH]
```

**Arguments**:

* `VOLUME_NAME`: \[required]
* `LOCAL_PATH`: \[required]
* `[REMOTE_PATH]`: \[default: /]

**Options**:

* `-f, --force`: Overwrite existing files.
* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

## `modal volume rm`

Delete a file or directory from a modal.Volume.

**Usage**:

```shell
modal volume rm [OPTIONS] VOLUME_NAME REMOTE_PATH
```

**Arguments**:

* `VOLUME_NAME`: \[required]
* `REMOTE_PATH`: \[required]

**Options**:

* `-r, --recursive`: Delete directory recursively
* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

## `modal volume cp`

Copy within a modal.Volume. Copy source file to destination file or multiple source files to destination directory.

**Usage**:

```shell
modal volume cp [OPTIONS] VOLUME_NAME PATHS...
```

**Arguments**:

* `VOLUME_NAME`: \[required]
* `PATHS...`: \[required]

**Options**:

* `-r, --recursive`: Copy directories recursively
* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

## `modal volume delete`

Delete a named Volume and all of its data.

**Usage**:

```shell
modal volume delete [OPTIONS] NAME
```

**Arguments**:

* `NAME`: Name of the modal.Volume to be deleted. Case sensitive  \[required]

**Options**:

* `--allow-missing`: Don't error if the Volume doesn't exist.
* `-y, --yes`: Run without pausing for confirmation.
* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.

## `modal volume rename`

Rename a modal.Volume.

**Usage**:

```shell
modal volume rename [OPTIONS] OLD_NAME NEW_NAME
```

**Arguments**:

* `OLD_NAME`: \[required]
* `NEW_NAME`: \[required]

**Options**:

* `-y, --yes`: Run without pausing for confirmation.
* `-e, --env TEXT`: Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
* `--help`: Show this message and exit.


This example shows how to run [WhisperX](https://github.com/m-bain/whisperX) on
Modal for accurate, word-level timestamped transcription.

We’ll walk through the following steps:

1. Defining the container image with CUDA 12.8, cuDNN 8, FFmpeg and Python deps.
2. Persisting model weights to a [Modal Volume](https://modal.com/docs/reference/modal.Volume).
3. A [Modal Cls](https://modal.com/docs/reference/modal.App#cls) that loads WhisperX once per GPU instance.
4. A [local entrypoint](https://modal.com/docs/reference/modal.App#local_entrypoint) that uploads an audio file to the service.

## Defining image

We start from NVIDIA’s official CUDA 12.8 devel image, add cuDNN, FFmpeg, and
install the WhisperX Python package plus its numerical deps.

```python
import os
import tempfile
from typing import Dict

import modal

MODEL_CACHE_DIR = "/whisperx-cache"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    # ── System deps ─────────────────────────────────────────────────────────────
    .apt_install("ffmpeg")  # audio decoding / resampling
    .apt_install("libcudnn8")  # cuDNN runtime
    .apt_install("libcudnn8-dev")  # cuDNN headers (needed by torch wheels)
    # ── Python deps ─────────────────────────────────────────────────────────────
    .pip_install(
        "whisperx==3.4.0",  # our ASR library
        "numpy==2.0.2",
        "scipy==1.15.0",
    )
    # Tell HF & Torch to cache inside our Volume
    .env({"HF_HOME": MODEL_CACHE_DIR})
    .env({"TORCH_HOME": MODEL_CACHE_DIR})
)

```

## Defining the app

Downloaded weights live in a [Modal Volume](https://modal.com/docs/reference/modal.Volume) so subsequent runs reuse them.

```python
app = modal.App("example-whisperx-transcribe", image=image)
models_volume = modal.Volume.from_name("whisperx-models", create_if_missing=True)

```

## Defining the inference service

We wrap WhisperX inference in a Modal Cls.
A single GPU container can serve multiple concurrent requests.

```python
@app.cls(
    gpu="H100",
    image=image,
    volumes={MODEL_CACHE_DIR: models_volume},
    timeout=30 * 60,
)
class WhisperX:
    """Serverless WhisperX service running on a single GPU."""

    @modal.enter()
    def setup(self):
        print("🔄 Loading WhisperX model …")
        import whisperx

        self.model = whisperx.load_model(
            "large-v2",
            device="cuda",
            compute_type="float16",
            download_root=MODEL_CACHE_DIR,
        )
        print("✅ Model ready!")

    @modal.method()
    def transcribe(self, audio_data: bytes) -> Dict:
        """
        Transcribe an audio file passed in as raw bytes.
        Returns language, per-word segments, and total duration.
        """

        import whisperx

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name

        try:
            audio = whisperx.load_audio(temp_audio_path)
            result = self.model.transcribe(audio, batch_size=16, language="en")

            language = result.get("language", "en")

            if result["segments"]:
                try:
                    align_model, metadata = whisperx.load_align_model(
                        language_code=language,
                        device=self.device,
                        model_dir=MODEL_CACHE_DIR,
                    )
                    result = whisperx.align(
                        result["segments"], align_model, metadata, audio, self.device
                    )
                except Exception as e:
                    print(f"⚠️ Alignment failed: {e} — falling back to segment-level")

            return {
                "language": language,
                "segments": result["segments"],
                "duration": len(audio) / 16_000,  # audio is 16 kHz
            }

        finally:
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

```

## Command-line usage

We expose a [local entrypoint](https://modal.com/docs/reference/modal.App#local_entrypoint)
so you can run:
- using a local audio file
- using a link to an audio file

```bash
modal run whisperx_transcribe.py --audio-file audio.wav # uses a local audio file
modal run whisperx_transcribe.py --audio-link https://example.com/audio.wav # uses a link to an audio file
modal run whisperx_transcribe.py # uses a default public audio file
```

```python
@app.local_entrypoint()
def main(
    audio_file: str = None,
    audio_link: str = None,
):
    import json
    import time

    import requests

    if not audio_file and not audio_link:
        print("No audio file or link provided, using default link")
        audio_link = "https://modal-public-assets.s3.us-east-1.amazonaws.com/erik.wav"

    if audio_file:
        print(f"🔊 Reading {audio_file} …")
        with open(audio_file, "rb") as f:
            audio_data = f.read()
    elif audio_link:
        print(f"🔊 Reading {audio_link} …")
        audio_data = requests.get(audio_link).content

    transcriber = WhisperX()

    print("📝 Transcribing …")
    start = time.time()
    result = transcriber.transcribe.remote(audio_data)
    duration = time.time() - start

    print(f"\n🌐 Detected language: {result['language']}")
    print(f"⏱️  Audio duration:   {result['duration']:.2f} s")
    print(f"🚀 Time taken:        {duration:.2f} s")

    with open("transcription.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n💾 Saved transcription → transcription.json")

```