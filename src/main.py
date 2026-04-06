import typer
from train import train
from chat import infer

app = typer.Typer()

app.command()(train)
app.command()(infer)

if __name__ == "__main__":
    app()