# Installing Software with Singularity

Brian DuSell<br>
Apr 9, 2019

Grad Tutorial Talk<br>
Dept. of Computer Science and Engineering<br>
University of Notre Dame

## Introduction

This tutorial will introduce you to [Singularity](https://www.sylabs.io/singularity/),
a containerization system for scientific computing environments that is
available on Notre Dame's CRC computing cluster. Containers allow you to
package the environment that your code depends on inside of a portable unit.
This is extremely useful for ensuring that your code can be run portably
on other machines. It is also usefuly for installing software, packages,
libraries, etc. in environments where you do not have root privileges, like the
CRC. I will show you how to install PyTorch with GPU support inside of a
container and run a simple PyTorch program to train a neural net.

## The Portability Problem

The programs we write depend on external environments, whether that environment
is explicitly documented or not. A Python program assumes that a Python
interpreter is available on the system it is run on. A Python program that uses
set comprehension syntax, e.g.

```python
{ x * 2 for x in range(10) }
```

assumes that you're using Python 3. A Python program that uses the function
`subprocess.run()` assumes that you're using at least version 3.5. A Python
program that calls `subprocess.run(['grep', '-r', 'foo', './my/directory'])`
assumes that you're running on a \*nix system where the program `grep` is
available.

When these dependencies are undocumented, it can become painful to run a
program in an environment that is different from the one it was developed in.
It would be nice to have a way to package a program together with its
environment, and then run that program on any machine.

## The Installation Problem

The CRC is a shared scientific computing environment with a shared file system.
This means that users do not have root privileges and cannot use a package
manager like `yum` or `apt-get` to install new libraries. If you want to
install something on the CRC that is not already there, you have a few options:

* If it is a major library, ask the staff to install/update it for you
* Install it in your home directory (e.g. `pip install --user` for Python
  modules) or other non-standard directory
* Compile it yourself in your home directory

While it is almost always possible to re-compile a library yourself without
root privileges, it can be very time-consuming. This is especially true when
the library depends on other libraries that also need to be re-compiled,
leading to a tedious search for just the right configuration to stitch them all
together. CUDA also complicates the situation, as certain deep learning
libraries need to be built on a node that has a GPU (even though the GPU is
never used during compilation!).

Finally, sometimes you deliberately want to install an older version of a
package. But unless you set up two isolated installations, this could conflict
with projects that still require the newer versions.

To take an extreme (but completely real!) example, older versions of the deep
learning library [DyNet](https://dynet.readthedocs.io/en/latest/) could only be
built with an old version of GCC, and moreover needed to be compiled on a GPU
node with the CRC's CUDA module loaded in order to work properly. In May 2018,
the CRC removed the required version of GCC. This meant that if you wanted to
install or update DyNet, you needed to re-compile that version of GCC yourself
*and* figure out how to configure DyNet to build itself with a compiler in a
non-standard location.

## The Solution: Containers

Containers are a software isolation technique that has exploded in popularity
in recent years, particularly thanks to [Docker](https://www.docker.com/).
A container, like a virtual machine, is an operating system within an operating
system. Unlike a virtual machine, however, it shares the kernel with the host
operating system, so it incurs no performance penalty for translating machine
instructions. Instead, containers rely on special system calls that allow the
host to spoof the filesystem and network that the container has access to,
making it appear from inside the container that it exists in a separate
environment.

Today we will be talking about an alternative to Docker called Singularity,
which is more suitable for scientific computing environments (Docker is better
suited for things like cloud applications, and there are reasons why it would
not be ideal for a shared environment like the CRC). The CRC currently offers
[Singularity 3.0](https://www.sylabs.io/guides/3.0/user-guide/), which is
available via the `singularity` command.

Singularity containers are instantiated from **images**, which are files that
define the container's environment. The container's "root" file system is
distinct from that of the host operating system, so you can install whatever
software you like as if you were the root user. Installing software via the
built-in package manager is now an option again. Not only this, but you can
also choose a pre-made image to base your container on. Singularity is
compatible with Docker images (a very deliberate design decision), so it can
take advantage of the extremely rich selection of production-grade Docker
images that are available. For example, there are pre-made images for fresh
installations of Ubuntu, Python, TensorFlow, PyTorch, and even CUDA. For
virtually all major libraries, getting a pre-made image for X is as simple as
Googling "X docker" and taking note of the name of the image.

Also, because your program's environment is self-contained, it is not affected
by changes to the CRC's software and is no longer susceptible to "software
rot." There is also no longer a need to rely on the CRC's modules via `module
load`. Because the container is portable, it will also run just as well on your
local machine as on the CRC. In the age of containers, "it runs on my machine"
is no longer an excuse.

## Basic Workflow

Singularity instantiates containers from images that define their environment.
Singularity images are stored in `.sif` files. You build a `.sif` file by
defining your environment in a text file and providing that definition to the
command `singularity build`.

Building an image file does require root privileges, so it is most convenient
to build the image on your local machine or workstation and then copy it to
your `/scratch365` directory in the CRC. The reason it requires root is because
the kernel is shared, and user permissions are implemented in the kernel. So if
you want to do something in the container as root, you actually need to *be*
root on the host when you do it.

There is also an option to build it without root privileges. This works by
sending your definition to a remote server and building the image there, but I
have had difficulty getting this to work.

Once you've uploaded your image to the CRC, you can submit a batch job that
runs `singularity exec` with the image file you created and the command you
want to run. That's it!

## A Simple PyTorch Program

I have included a PyTorch program,
[`train_xor.py`](examples/xor/train_xor.py),
that trains a neural network to compute the XOR function and then plots the
loss as a function of training time. It can also save the model to a file. It
depends on the Python modules `torch`, `numpy`, and `matplotlib`.

## Installing Singularity

[Singularity 3.0](https://www.sylabs.io/guides/3.0/user-guide/index.html)
is already available on the CRC via the `singularity` command.

As for installing Singularity locally, the Singularity docs include detailed
instructions for installing Singularity on major operating systems
[here](https://www.sylabs.io/guides/3.0/user-guide/installation.html).
Installing Singularity is not necessary for following the tutorial in real
time, as I will provide you with pre-built images.

## Defining an Image

The first step in defining an image is picking which base image to use. This
can be a Linux distribution, such as Ubuntu, or an image with a library
pre-installed, like one of PyTorch's
[official Docker images](https://hub.docker.com/r/pytorch/pytorch/tags). Since
our program depends on more than just PyTorch, let's start with a plain Ubuntu
image and build up from there.

Let's start with the basic syntax for definition files, which is documented
[here](https://www.sylabs.io/guides/3.0/user-guide/definition_files.html).
The first part of the file is the header, where we define the base image and
other meta-information. The only required keyword in the header is `Bootstrap`,
which defines the type of image being imported. Using `Bootstrap: library`
means that we are importing a library from the official
[Singularity Library](https://cloud.sylabs.io/library).
Using `Bootstrap: docker` means that we are importing a Docker image from a
Docker registry such as
[Docker Hub](https://hub.docker.com/).
Let's import the official
[Ubuntu 18.04](https://cloud.sylabs.io/library/_container/5baba99394feb900016ea433)
image.

```
Bootstrap: library
From: ubuntu:18.04
```

The rest of the definition file is split up into several **sections** which
serve special roles. The `%post` section defines a series of commands to be run
while the image is being built, inside of a container as the root user. This
is typically where you install packages. The `%environment` section defines
environment variables that are set when the image is instantiated as a
container. The `%files` section lets you copy files into the image. There are
[many other types of section](https://www.sylabs.io/guides/3.0/user-guide/definition_files.html#sections).

Let's use the `%post` section to install all of our requirements using
`apt-get` and `pip3`. 

```
%post
    # Downloads the latest package lists (important).
    apt-get update -y
    # Runs apt-get while ensuring that there are no user prompts that would
    # cause the build process to hang.
    # python3-tk is required by matplotlib.
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 \
        python3-tk \
        python3-pip
    # Reduce the size of the image by deleting the package lists we downloaded,
    # which are no longer needed.
    rm -rf /var/lib/apt/lists/*
    # Install Python modules.
    pip3 install torch numpy matplotlib
```

Each line defines a separate command (lines can be continued with a `\`).
Unlike normal shell scripts, the build will be aborted as soon as one of the
commands fails. You do not need to connect the commands with `&&`.

The final build definition is in the file
[version-1.def](examples/xor/version-1.def).

## Building an Image

Supposing we are on our own Ubuntu machine, we can build this definition into
a `.sif` image file using the following command:

```bash
cd examples/xor
sudo singularity build version-1.sif version-1.def
```

[View the screencast](https://bdusell.github.io/singularity-tutorial/casts/version-1.html)

This ran the commands we defined in the `%post` section inside a container and
afterwards saved the state of the container in the image `version-1.sif`.

## Running an Image

Let's run our PyTorch program in a container based on the image we just built.

```bash
singularity exec version-1.sif python3 train_xor.py --output model.pt
```

This program does not take long to run. Once it finishes, it should open a
window with a plot of the model's loss and accuracy over time.

[![asciicast](https://asciinema.org/a/Lqq0AsJSwVgFoo1Hr8S7euMe5.svg)](https://asciinema.org/a/Lqq0AsJSwVgFoo1Hr8S7euMe5)

![Plot](images/plot.png)

The trained model should also be saved in the file `model.pt`. Note that even
though the program ran in a container, it was able to write a file to the host
file system that remained after the program exited and the container was shut
down. If you are familiar with Docker, you probably know that you cannot write
files to the host in this way unless you explicitly **bind mount** two
directories in the host and container file system. Bind mounting makes a file
or directory on the host system synonymous with one in the conatiner.

For convenience, Singularity
[binds a few important directories by
default](https://www.sylabs.io/guides/3.0/user-guide/bind_paths_and_mounts.html):

* Your home directory
* The current working directory
* `/tmp`
* `/proc`
* `/sys`
* `/dev`

You can add to or override these settings if you wish using the
[`--bind` flag](https://www.sylabs.io/guides/3.0/user-guide/bind_paths_and_mounts.html#specifying-bind-paths)
to `singularity exec`. This is important to remember if you want to access a
file that is outside of your home directory on the CRC -- otherwise you may end
up with cryptic persmission errors.

It is also important to know that, unlike Docker, environment variables are
inherited inside the container for convenience.

## Running an Interactive Shell

You can also open up a shell inside the container and run commands there. You
can `exit` when you're done. Note that since your home directory is
bind-mounted, the shell inside the container will run your shell's startup file
(e.g. `.bashrc`).

```
$ singularity shell version-1.sif
Singularity version-1.sif:~/singularity-tutorial/examples/xor> python3 train_xor.py
```

## Running an Image on the CRC

* TODO scratch365
* TODO remember to bind directories outside of your home directory, or you'll get
  permission denied errors

## Adding GPU Support

* make sure to respect `CUDA_VISIBLE_DEVICES`

Maybe:

We will be running the official [`word_language_model`](word_language_model/)
example PyTorch program. This program trains an
[LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)
[language model](https://en.wikipedia.org/wiki/Language_model)
on a corpus of Wikipedia text.

* if there's time, Docker
