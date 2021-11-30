# Terminal Commands for Video Contrastive [Sapienza]

### Check Containers Run from an Image

```sh
docker ps -a -f ancestor=paolomandica/sapienza-video-contrastive `# ancestor identifies the image`
```

See also docs for `docker ps` listing containers <https://docs.docker.com/engine/reference/commandline/ps/>. 

### Docker Container

NB Before running, ensure the Docker is using the appropriate context (i.e. DGXContext locally). The context can be set with: `docker context use DGXContext`.

Run a container (with a specific name, replacing `NAME_OF_CONTAINER`.

Note that:

- ports 8092-8096 are unassigned, so can be allocated. See <https://www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.xhtml?&page=109>.
- When creating a new container, whilst the necessary packages can be installed with `pip install -r requirements.txt` from the project root, some including wandb install into `/home/francolu/.local/bin`. You can optionally add this to the `PATH` via

```sh
export PATH="$HOME/.local/bin:$PATH"

# or
export PATH="/home/francolu/.local/bin:$PATH"

# or
export PATH="/any/arbitrary/path/to/binaries/or/else/here:$PATH"
```
- Check the PATH environment variable (a colon-separated string) has been correctly updated with `echo $PATH$`. It should have the new path prepended (with a trailing colon).

### DGX Version

```sh
docker run -it \
--user "$(id -u):$(id -g)" \
--gpus '"device=2"' \
--name "NAME_OF_CONTAINER" \
--shm-size 8G \
-v /raid/data/francolu:/data_volume \
-p 8093:8093 \
paolomandica/video-simsiam-dgx
```

### PINLab Workstation Version

An amended version of the above with an appropriate mapping for the data location on PINLab's workstation. 

```sh
docker run -it \
--gpus '"device=0"' \
--name "NAME_OF_CONTAINER" \
--shm-size 8G \
-v /home/ares/luca/panasonic:/data_volume \
-p 8093:8093 \
paolomandica/video-simsiam
``` 

### Everything I Do After Creating a New Container

Run the following when attaching to a new container to clone this repo, `cd` into it, install required packages, add necessary directories to the PATH (e.g. for invoking W&B from the command line for login), reminding you which branch you are on (`main`) and then bringing up a prompt to get you to log into W&B. 

```sh
git clone https://github.com/paolomandica/video-simsiam.git && \
cd video-simsiam && \
pip install -r requirements.txt && \
export PATH="$HOME/.local/bin:$PATH" && \
git branch && \
wandb login
```

### Transferring Files via Secure Copy

Files can be transferred to paths on servers via the SSH protocol using `scp` (secure copy). An example command is shown below, specifying a local source and server path destination. 

```sh
scp .\checkpoint.pth francolu@sapienzaAI-01.roma1.infn.it:/raid/data/francolu/temporary_storage/sp-unnorm/
```

### Server Utilisation & Process Viewing

Watch (and update) GPU usage statistics on the server. 

NB `watch` with `-d` flag highlights diffs across prints, `-n` sets refresh rate. 

```sh
watch -d -n 0.5 nvidia-smi
```

Meanwhile, processes can be interactively viewed and sorted via the [`htop`](https://linux.die.net/man/1/htop) command line tool. 

## Attaching a Container to VSCode

You can attach VSCode to a running container. In this case, settings such as the `workspaceFolder` can be set as desired to e.g. attach the VSCode Explorer pane to the correct directory as the project root. 

Example of Sapienza Video Contrastive JSON:

```json
{
	"workspaceFolder": "/home/francolu"
}
```

Example generic `devcontainer.json`

```json
{
  // Default path to open when attaching to a new container.
  "workspaceFolder": "/path/to/code/in/container/here",

  // An array of extension IDs that specify the extensions to
  // install inside the container when you first attach to it.
  "extensions": ["dbaeumer.vscode-eslint"],

  // Any *default* container specific VS Code settings
  "settings": {
    "terminal.integrated.shell.linux": "/bin/bash"
  },

  // An array port numbers to forward
  "forwardPorts": [8000],

  // Container user VS Code should use when connecting
  "remoteUser": "vscode",

  // Set environment variables for VS Code and sub-processes
  "remoteEnv": { "MY_VARIABLE": "some-value" }
}
```

See <https://code.visualstudio.com/docs/remote/attach-container#_attached-container-configuration-files> for details. 

In particular very usefully how to edit dev container configuration files if something goes wrong:

> Tip: If something is wrong with your configuration, you can also edit it when not attached to the container by selecting Remote-Containers: Open Attached Container Configuration File... from the Command Palette (F1) and then picking the image / container name from the presented list.

NB The dev container configuration files on my machine are located in `C:\Users\anilk\AppData\Roaming\Code\User\globalStorage\ms-vscode-remote.remote-containers\imageConfigs`. 

## Logging with Weights & Biases (wandb)

Logging with Weights & Biases is done by logging in on a host machine, be it a local machine, the DGX server or any other server. 

After installing `wandb`, for example via `pip install wandb` as above, one can access tools via the command-line interface. For details, see `wandb --help`, which summarily lists the following commands

```
Commands:
  agent         Run the W&B agent
  artifact      Commands for interacting with artifacts
  controller    Run the W&B local sweep controller
  disabled      Disable W&B.
  docker        W&B docker lets you run your code in a docker image...
  docker-run    Simple wrapper for `docker run` which adds WANDB_API_KEY...
  enabled       Enable W&B.
  init          Configure a directory with Weights & Biases
  launch        Launch or queue a job from a uri (Experimental).
  launch-agent  Run a W&B launch agent (Experimental)
  local         Launch local W&B container (Experimental)
  login         Login to Weights & Biases
  offline       Disable W&B sync
  online        Enable W&B sync
  pull          Pull files from Weights & Biases
  restore       Restore code, config and docker state for a run
  status        Show configuration settings
  sweep         Create a sweep
  sync          Upload an offline training directory to W&B
  verify        Verify your local instance
```

Login can be done with

If already logged in, this will prompt display of the current logged in user, for example

```
wandb: Currently logged in as: anilkeshwani (use `wandb login --relogin` to force relogin)
```

If directly using wandb during training without prior authentication, a prompt will allow you to choose from

```
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
```

Upon logging in by pasting an API key, this will be appended to your netrc file, for example

```
Appending key for api.wandb.ai to your netrc file: /home/francolu/.netrc
```

Use `cat /home/francolu/.netrc` or generically `cat /path/to/.netrc` to view the contents, which initially are like

```
machine api.wandb.ai
  login user
  password [REDACTED; a hash-like string, not your plaintext password]
```

## screen

Linux `screen` offers the possibility of running virtual terminals inside a multiplexer and brings the advantage that remote session disconnects (e.g. SSH disconnects) do not stop the process running inside the virtual terminals. 

You start a screen session via `screen` and can "detach" from the session via `ctrl + a, d`. Then return to the detached session with `screen -r` or `screen -r [session_id]` when running multiple screens. NB list the running screen sessions with `screen -ls`. 

Create a new terminal (e.g. bash) with `ctrl + a, c` and list the open terminals in a screen with `ctrl + a, "`. 

More details are available at <https://linuxize.com/post/how-to-use-linux-screen/>. 

For more shortcut keys, see <http://www.pixelbeat.org/lkdb/screen.html> and consider also [tmux](https://github.com/tmux/tmux/wiki) as an alternative multiplexer. 
