"""Type stubs for docker library."""

import builtins
import datetime
from collections.abc import Iterator
from typing import Any

class DockerClient:
    """Docker client."""

    containers: ContainerCollection
    images: ImageCollection
    networks: NetworkCollection
    volumes: VolumeCollection

    def __init__(
        self,
        base_url: str | None = None,
        version: str | None = None,
        timeout: int | None = None,
        tls: bool | None = None,
        user_agent: str | None = None,
        **kwargs: Any,
    ) -> None: ...
    def ping(self) -> bool: ...
    def version(self) -> dict[str, Any]: ...
    def info(self) -> dict[str, Any]: ...
    def close(self) -> None: ...
    @classmethod
    def from_env(cls, **kwargs: Any) -> DockerClient: ...

class ContainerCollection:
    """Container collection."""

    def run(
        self,
        image: str | Image,
        command: str | builtins.list[str] | None = None,
        *,
        auto_remove: bool = False,
        detach: bool = False,
        environment: dict[str, str] | None = None,
        labels: dict[str, str] | None = None,
        name: str | None = None,
        network: str | None = None,
        ports: dict[str, Any] | None = None,
        remove: bool = False,
        restart_policy: dict[str, Any] | None = None,
        volumes: dict[str, dict[str, str]] | None = None,
        working_dir: str | None = None,
        **kwargs: Any,
    ) -> Container | bytes: ...
    def create(
        self,
        image: str | Image,
        command: str | builtins.list[str] | None = None,
        **kwargs: Any,
    ) -> Container: ...
    def get(self, container_id: str) -> Container: ...
    def list(
        self,
        all: bool = False,
        before: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
        since: str | None = None,
        sparse: bool = False,
        ignore_removed: bool = False,
    ) -> builtins.list[Container]: ...
    def prune(self, filters: dict[str, Any] | None = None) -> dict[str, Any]: ...

class Container:
    """Docker container."""

    id: str
    short_id: str
    name: str
    status: str
    attrs: dict[str, Any]
    image: Image
    labels: dict[str, str]

    def attach(
        self,
        stdout: bool = True,
        stderr: bool = True,
        stream: bool = False,
        logs: bool = False,
        **kwargs: Any,
    ) -> str | Iterator[str]: ...
    def exec_run(
        self,
        cmd: str | list[str],
        stdout: bool = True,
        stderr: bool = True,
        stdin: bool = False,
        tty: bool = False,
        privileged: bool = False,
        user: str | None = None,
        detach: bool = False,
        stream: bool = False,
        socket: bool = False,
        environment: dict[str, str] | None = None,
        workdir: str | None = None,
        demux: bool = False,
    ) -> tuple[int, bytes | Iterator[bytes]] | ExecResult: ...
    def kill(self, signal: str | int | None = None) -> None: ...
    def logs(
        self,
        stdout: bool = True,
        stderr: bool = True,
        stream: bool = False,
        timestamps: bool = False,
        tail: str | int | None = None,
        since: datetime.datetime | int | None = None,
        follow: bool | None = None,
        until: datetime.datetime | int | None = None,
    ) -> str | Iterator[str]: ...
    def pause(self) -> None: ...
    def remove(
        self, v: bool = False, link: bool = False, force: bool = False
    ) -> None: ...
    def rename(self, name: str) -> None: ...
    def restart(self, timeout: int | None = None) -> None: ...
    def start(self, **kwargs: Any) -> None: ...
    def stats(
        self, decode: bool = False, stream: bool = True
    ) -> dict[str, Any] | Iterator[dict[str, Any]]: ...
    def stop(self, timeout: int | None = None) -> None: ...
    def top(self, ps_args: str | None = None) -> dict[str, Any]: ...
    def unpause(self) -> None: ...
    def update(self, **kwargs: Any) -> dict[str, Any]: ...
    def wait(
        self, timeout: int | None = None, condition: str | None = None
    ) -> dict[str, Any]: ...
    def reload(self) -> None: ...

class ImageCollection:
    """Image collection."""

    def build(
        self,
        path: str | None = None,
        fileobj: Any | None = None,
        tag: str | None = None,
        quiet: bool = False,
        nocache: bool = False,
        rm: bool = False,
        timeout: int | None = None,
        dockerfile: str | None = None,
        buildargs: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[Image, Iterator[dict[str, Any]]]: ...
    def get(self, name: str) -> Image: ...
    def list(
        self,
        name: str | None = None,
        all: bool = False,
        filters: dict[str, Any] | None = None,
    ) -> builtins.list[Image]: ...
    def load(self, data: bytes | Any) -> builtins.list[Image]: ...
    def pull(
        self,
        repository: str,
        tag: str | None = None,
        all_tags: bool = False,
        **kwargs: Any,
    ) -> Image | builtins.list[Image]: ...
    def push(
        self,
        repository: str,
        tag: str | None = None,
        stream: bool = False,
        auth_config: dict[str, str] | None = None,
        decode: bool = False,
    ) -> str | Iterator[str]: ...
    def remove(
        self, image: str | Image, force: bool = False, noprune: bool = False
    ) -> None: ...
    def search(
        self, term: str, limit: int | None = None
    ) -> builtins.list[dict[str, Any]]: ...
    def prune(self, filters: dict[str, Any] | None = None) -> dict[str, Any]: ...

class Image:
    """Docker image."""

    id: str
    short_id: str
    tags: list[str]
    attrs: dict[str, Any]

    def history(self) -> list[dict[str, Any]]: ...
    def reload(self) -> None: ...
    def save(
        self, chunk_size: int = 2097152, named: bool = False
    ) -> Iterator[bytes]: ...
    def tag(
        self, repository: str, tag: str | None = None, force: bool = False
    ) -> bool: ...

class NetworkCollection:
    """Network collection."""

    def create(
        self,
        name: str,
        driver: str | None = None,
        options: dict[str, Any] | None = None,
        ipam: dict[str, Any] | None = None,
        check_duplicate: bool | None = None,
        internal: bool = False,
        labels: dict[str, str] | None = None,
        enable_ipv6: bool = False,
        attachable: bool | None = None,
        scope: str | None = None,
        ingress: bool | None = None,
    ) -> Network: ...
    def get(
        self, network_id: str, verbose: bool | None = None, scope: str | None = None
    ) -> Network: ...
    def list(
        self,
        names: builtins.list[str] | None = None,
        ids: builtins.list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> builtins.list[Network]: ...
    def prune(self, filters: dict[str, Any] | None = None) -> dict[str, Any]: ...

class Network:
    """Docker network."""

    id: str
    short_id: str
    name: str
    containers: list[Container]
    attrs: dict[str, Any]

    def connect(
        self,
        container: str | Container,
        aliases: list[str] | None = None,
        links: list[tuple[str, str]] | None = None,
        ipv4_address: str | None = None,
        ipv6_address: str | None = None,
        link_local_ips: list[str] | None = None,
    ) -> None: ...
    def disconnect(self, container: str | Container, force: bool = False) -> None: ...
    def reload(self) -> None: ...
    def remove(self) -> None: ...

class VolumeCollection:
    """Volume collection."""

    def create(
        self,
        name: str,
        driver: str | None = None,
        driver_opts: dict[str, Any] | None = None,
        labels: dict[str, str] | None = None,
    ) -> Volume: ...
    def get(self, volume_id: str) -> Volume: ...
    def list(self, filters: dict[str, Any] | None = None) -> builtins.list[Volume]: ...
    def prune(self, filters: dict[str, Any] | None = None) -> dict[str, Any]: ...

class Volume:
    """Docker volume."""

    id: str
    short_id: str
    name: str
    attrs: dict[str, Any]

    def reload(self) -> None: ...
    def remove(self, force: bool = False) -> None: ...

class ExecResult:
    """Result of exec_run."""

    exit_code: int
    output: (
        bytes | tuple[bytes, bytes] | Iterator[bytes] | Iterator[tuple[bytes, bytes]]
    )

def from_env(**kwargs: Any) -> DockerClient:
    """Create a client from environment variables."""
