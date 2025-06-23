"""Type stubs for docker library."""

from typing import Any, Dict, List, Optional, Union, Iterator
import datetime

class DockerClient:
    """Docker client."""
    
    containers: "ContainerCollection"
    images: "ImageCollection"
    networks: "NetworkCollection"
    volumes: "VolumeCollection"
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        version: Optional[str] = None,
        timeout: Optional[int] = None,
        tls: Optional[bool] = None,
        user_agent: Optional[str] = None,
        **kwargs: Any
    ) -> None: ...
    
    def ping(self) -> bool: ...
    def version(self) -> Dict[str, Any]: ...
    def info(self) -> Dict[str, Any]: ...
    def close(self) -> None: ...
    
    @classmethod
    def from_env(cls, **kwargs: Any) -> "DockerClient": ...

class ContainerCollection:
    """Container collection."""
    
    def run(
        self,
        image: Union[str, "Image"],
        command: Optional[Union[str, List[str]]] = None,
        *,
        auto_remove: bool = False,
        detach: bool = False,
        environment: Optional[Dict[str, str]] = None,
        labels: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        network: Optional[str] = None,
        ports: Optional[Dict[str, Any]] = None,
        remove: bool = False,
        restart_policy: Optional[Dict[str, Any]] = None,
        volumes: Optional[Dict[str, Dict[str, str]]] = None,
        working_dir: Optional[str] = None,
        **kwargs: Any
    ) -> Union["Container", bytes]: ...
    
    def create(
        self,
        image: Union[str, "Image"],
        command: Optional[Union[str, List[str]]] = None,
        **kwargs: Any
    ) -> "Container": ...
    
    def get(self, container_id: str) -> "Container": ...
    
    def list(
        self,
        all: bool = False,
        before: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        since: Optional[str] = None,
        sparse: bool = False,
        ignore_removed: bool = False,
    ) -> List["Container"]: ...
    
    def prune(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...

class Container:
    """Docker container."""
    
    id: str
    short_id: str
    name: str
    status: str
    attrs: Dict[str, Any]
    image: "Image"
    labels: Dict[str, str]
    
    def attach(
        self,
        stdout: bool = True,
        stderr: bool = True,
        stream: bool = False,
        logs: bool = False,
        **kwargs: Any
    ) -> Union[str, Iterator[str]]: ...
    
    def exec_run(
        self,
        cmd: Union[str, List[str]],
        stdout: bool = True,
        stderr: bool = True,
        stdin: bool = False,
        tty: bool = False,
        privileged: bool = False,
        user: Optional[str] = None,
        detach: bool = False,
        stream: bool = False,
        socket: bool = False,
        environment: Optional[Dict[str, str]] = None,
        workdir: Optional[str] = None,
        demux: bool = False,
    ) -> Union[tuple[int, Union[bytes, Iterator[bytes]]], "ExecResult"]: ...
    
    def kill(self, signal: Optional[Union[str, int]] = None) -> None: ...
    def logs(
        self,
        stdout: bool = True,
        stderr: bool = True,
        stream: bool = False,
        timestamps: bool = False,
        tail: Optional[Union[str, int]] = None,
        since: Optional[Union[datetime.datetime, int]] = None,
        follow: Optional[bool] = None,
        until: Optional[Union[datetime.datetime, int]] = None,
    ) -> Union[str, Iterator[str]]: ...
    def pause(self) -> None: ...
    def remove(self, v: bool = False, link: bool = False, force: bool = False) -> None: ...
    def rename(self, name: str) -> None: ...
    def restart(self, timeout: Optional[int] = None) -> None: ...
    def start(self, **kwargs: Any) -> None: ...
    def stats(self, decode: bool = False, stream: bool = True) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]: ...
    def stop(self, timeout: Optional[int] = None) -> None: ...
    def top(self, ps_args: Optional[str] = None) -> Dict[str, Any]: ...
    def unpause(self) -> None: ...
    def update(self, **kwargs: Any) -> Dict[str, Any]: ...
    def wait(self, timeout: Optional[int] = None, condition: Optional[str] = None) -> Dict[str, Any]: ...
    def reload(self) -> None: ...

class ImageCollection:
    """Image collection."""
    
    def build(
        self,
        path: Optional[str] = None,
        fileobj: Optional[Any] = None,
        tag: Optional[str] = None,
        quiet: bool = False,
        nocache: bool = False,
        rm: bool = False,
        timeout: Optional[int] = None,
        dockerfile: Optional[str] = None,
        buildargs: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> tuple["Image", Iterator[Dict[str, Any]]]: ...
    
    def get(self, name: str) -> "Image": ...
    
    def list(
        self,
        name: Optional[str] = None,
        all: bool = False,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List["Image"]: ...
    
    def load(self, data: Union[bytes, Any]) -> List["Image"]: ...
    def pull(
        self,
        repository: str,
        tag: Optional[str] = None,
        all_tags: bool = False,
        **kwargs: Any
    ) -> Union["Image", List["Image"]]: ...
    def push(
        self,
        repository: str,
        tag: Optional[str] = None,
        stream: bool = False,
        auth_config: Optional[Dict[str, str]] = None,
        decode: bool = False,
    ) -> Union[str, Iterator[str]]: ...
    def remove(self, image: Union[str, "Image"], force: bool = False, noprune: bool = False) -> None: ...
    def search(self, term: str, limit: Optional[int] = None) -> List[Dict[str, Any]]: ...
    def prune(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...

class Image:
    """Docker image."""
    
    id: str
    short_id: str
    tags: List[str]
    attrs: Dict[str, Any]
    
    def history(self) -> List[Dict[str, Any]]: ...
    def reload(self) -> None: ...
    def save(self, chunk_size: int = 2097152, named: bool = False) -> Iterator[bytes]: ...
    def tag(self, repository: str, tag: Optional[str] = None, force: bool = False) -> bool: ...

class NetworkCollection:
    """Network collection."""
    
    def create(
        self,
        name: str,
        driver: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        ipam: Optional[Dict[str, Any]] = None,
        check_duplicate: Optional[bool] = None,
        internal: bool = False,
        labels: Optional[Dict[str, str]] = None,
        enable_ipv6: bool = False,
        attachable: Optional[bool] = None,
        scope: Optional[str] = None,
        ingress: Optional[bool] = None,
    ) -> "Network": ...
    
    def get(self, network_id: str, verbose: Optional[bool] = None, scope: Optional[str] = None) -> "Network": ...
    
    def list(self, names: Optional[List[str]] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, Any]] = None) -> List["Network"]: ...
    
    def prune(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...

class Network:
    """Docker network."""
    
    id: str
    short_id: str
    name: str
    containers: List[Container]
    attrs: Dict[str, Any]
    
    def connect(self, container: Union[str, Container], aliases: Optional[List[str]] = None, links: Optional[List[tuple[str, str]]] = None, ipv4_address: Optional[str] = None, ipv6_address: Optional[str] = None, link_local_ips: Optional[List[str]] = None) -> None: ...
    def disconnect(self, container: Union[str, Container], force: bool = False) -> None: ...
    def reload(self) -> None: ...
    def remove(self) -> None: ...

class VolumeCollection:
    """Volume collection."""
    
    def create(
        self,
        name: str,
        driver: Optional[str] = None,
        driver_opts: Optional[Dict[str, Any]] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> "Volume": ...
    
    def get(self, volume_id: str) -> "Volume": ...
    
    def list(self, filters: Optional[Dict[str, Any]] = None) -> List["Volume"]: ...
    
    def prune(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...

class Volume:
    """Docker volume."""
    
    id: str
    short_id: str
    name: str
    attrs: Dict[str, Any]
    
    def reload(self) -> None: ...
    def remove(self, force: bool = False) -> None: ...

class ExecResult:
    """Result of exec_run."""
    exit_code: int
    output: Union[bytes, tuple[bytes, bytes], Iterator[bytes], Iterator[tuple[bytes, bytes]]]

def from_env(**kwargs: Any) -> DockerClient:
    """Create a client from environment variables."""
    ...