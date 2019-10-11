import click

@click.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.option(
    "-u",
    "--username",
    type=str,
    default="Gangus"
)
@click.option(
    "-p",
    "--port",
    type=int,
    default=8100
)
@click.option(
    "-h",
    "--host",
    type=str,
    default="discovery"
)
@click.option(
    "-t",
    "--tunnel_host",
    type=str,
    default="myth.stanford.edu"
)
def connect(username, host, port, tunnel_host):
    """
    """
    host_addr = f"{username}@{host}"
    tunnel_username = username if username == "Gangus" else "sabrieyuboglu"
    tunnel_addr = f"{tunnel_username}@{tunnel_host}"

    print(f"Forwarding: {tunnel_addr} -> local")
    os.system(f"ssh -N -f -L localhost:{port}:localhost:{port} {tunnel_addr}")
    print("----------------------------")
    print(f"Forwarding: {host_addr} -> {tunnel_addr}")
    os.system(f"ssh -t {tunnel_addr} 'ssh -N -L localhost:{port}:localhost:{port} {host_addr}'")
    print("----------------------------")
    print(f"Terminating connection to: {host} port {port}...")
