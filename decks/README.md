# Training Decks

Place `.dck` deck files here for use with training scripts and the card recommender.

This directory is **gitignored** — deck files are local only. Use the migration script to transfer decks between machines:

```bash
./scripts/migrate_claude_code.sh --export   # bundles decks/ into tarball
./scripts/migrate_claude_code.sh --import   # extracts decks/ on new machine
```

## Deck Format

Forge `.dck` format:

```
[metadata]
Name=Deck Name

[Main]
4 Lightning Bolt
4 Monastery Swiftspear
20 Mountain

[Sideboard]
2 Searing Blood
```

## Deck Sources

- **Forge**: Export from Forge deck editor
- **MTGGoldfish**: `scripts/scrape_mtggoldfish.py` outputs `.dck` format
- **Manual**: Any text file with `N Card Name` per line
