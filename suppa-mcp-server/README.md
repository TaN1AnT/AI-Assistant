# suppa-mcp-server

MCP (Model Context Protocol) server for the **Suppa** backend service.  
Allows LLMs (Claude, GPT, Copilot, Cursor, etc.) to interact with the Suppa platform through standardised MCP tools.

[![npm version](https://img.shields.io/npm/v/suppa-mcp-server)](https://www.npmjs.com/package/suppa-mcp-server)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Quick start

### Install from npm

```bash
npm install -g suppa-mcp-server
```

### Or run directly with npx

```bash
SUPPA_API_KEY=your_key npx suppa-mcp-server
```

---

## Configuration

| Variable | Required | Description |
|---|---|---|
| `SUPPA_API_KEY` | ✅ | Bearer token for the Suppa API |
| `SUPPA_API_URL` | ❌ | Override the base API URL (default `https://sp.modern-expo.com`) |
| `TRANSPORT` | ❌ | `stdio` (default) or `http` |
| `PORT` | ❌ | HTTP port when `TRANSPORT=http` (default `3000`) |

---

## Use with Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "suppa": {
      "command": "npx",
      "args": ["-y", "suppa-mcp-server"],
      "env": {
        "SUPPA_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

## Use with Cursor

Add to `.cursor/mcp.json` in your project:

```json
{
  "mcpServers": {
    "suppa": {
      "command": "npx",
      "args": ["-y", "suppa-mcp-server"],
      "env": {
        "SUPPA_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

## Use with VS Code (Copilot)

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "suppa": {
      "command": "npx",
      "args": ["-y", "suppa-mcp-server"],
      "env": {
        "SUPPA_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

---

## Agent workflow

When an AI agent connects to this MCP server, it receives built-in instructions describing the correct workflow. The server enforces a **discovery-first** approach:

```
┌─────────────────────────────────────────────────────────┐
│  1. DISCOVER ENTITIES                                   │
│     suppa_list_entities(search?: "task")                │
│     → Get entity IDs and aliases                        │
├─────────────────────────────────────────────────────────┤
│  2. DISCOVER FIELDS                                     │
│     suppa_get_entity_props(entity_id: "...")             │
│     → Get field names, types, comparators, relations    │
├─────────────────────────────────────────────────────────┤
│  3. READ DATA                                           │
│     suppa_search_instances   — search with filters      │
│     suppa_get_instance       — get single record        │
│     suppa_get_child_instances— get sub-records          │
│     suppa_get_comments       — get comments             │
│     suppa_get_mentions       — get unread @mentions     │
│     suppa_get_custom_enum_values — get enum options     │
├─────────────────────────────────────────────────────────┤
│  4. WRITE DATA                                          │
│     suppa_create_instance    — create record            │
│     suppa_update_instance    — update record            │
│     suppa_create_comment     — add comment (@mentions)  │
└─────────────────────────────────────────────────────────┘
```

> **Important:** Agents must always call `suppa_list_entities` → `suppa_get_entity_props` before reading or writing data. Field names, types, and available filters vary between entities.

---

## Available tools

| Tool | Description |
|---|---|
| `suppa_list_entities` | List all available entities (tables) with optional name search |
| `suppa_get_entity_props` | Get field definitions and types for a given entity |
| `suppa_search_instances` | Search / filter records in any entity with pagination |
| `suppa_get_instance` | Get a single record by entity ID and instance ID |
| `suppa_get_child_instances` | Get child records (subtasks, sub-items) by parent ID |
| `suppa_get_comments` | Get comments on a specific record |
| `suppa_get_mentions` | Get records with unread @mentions for the current user |
| `suppa_create_comment` | Add a comment to a record (supports @mentions) |
| `suppa_create_instance` | Create a new record in any entity |
| `suppa_update_instance` | Update an existing record (partial update — only changed fields) |
| `suppa_get_custom_enum_values` | Get available values for custom enum fields |

### Special entity aliases

- `SmartUser` — Search and manage users
- Any entity UUID — Access any custom table

---

## Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Run locally (stdio)
SUPPA_API_KEY=test node dist/index.js

# Run locally (HTTP)
SUPPA_API_KEY=test TRANSPORT=http node dist/index.js

# Test with MCP Inspector
npm run inspect

# Watch mode
npm run dev
```

---

## Publish to npm

```bash
# 1. Make sure you're logged in
npm login

# 2. Build & publish
npm publish
```

The `prepublishOnly` script will automatically run `tsc` before publishing.

---

## Project structure

```
suppa-mcp-server/
├── src/
│   ├── index.ts                # Entry point — McpServer + transport
│   ├── constants.ts            # API URL, limits, prefix
│   ├── schemas/
│   │   └── common.ts           # Reusable Zod schemas
│   ├── services/
│   │   └── apiClient.ts        # Authenticated HTTP client
│   └── tools/
│       ├── index.ts            # Tool registry
│       ├── listEntities.ts     # List all entities
│       ├── getEntityProps.ts   # Get entity field definitions
│       ├── searchInstances.ts  # Search records
│       ├── getInstance.ts      # Get single record
│       ├── getChildInstances.ts# Get child records
│       ├── getComments.ts      # Get comments
│       ├── getMentions.ts     # Get unread @mentions
│       ├── createComment.ts   # Create comment
│       ├── createInstance.ts   # Create record
│       ├── updateInstance.ts   # Update record
│       └── getCustomEnumValues.ts # Get enum values
├── package.json
├── tsconfig.json
├── vitest.config.ts
└── README.md
```

---

## License

MIT

