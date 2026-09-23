# gacha

Exact waiting-time models for gacha banners, starting with the Arknights: Endfield limited
character banner. See `docs/superpowers/specs/` for the design and `CLAUDE.md` for conventions.

    uv sync
    uv run pytest
    uv run gacha evaluate --pity 40 --banner-pulls 20 --budget 60
    uv run gacha experiment all
