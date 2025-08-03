# TiddlyWiki Architecture - Minimal Spanning Graph

```mermaid
graph TD
    %% Core Boot System
    BOOT[boot/boot.js] --> WIKI[core/modules/wiki.js]
    BOOT --> WIDGET[core/modules/widgets/widget.js]
    
    %% Wiki Engine - Central Hub
    WIKI --> TIDDLER[core/modules/tiddler.js]
    WIKI --> PARSER[core/modules/parsers/]
    WIKI --> FILTER[core/modules/filters/]
    WIKI --> MACRO[core/modules/macros/]
    
    %% Widget System - UI Layer
    WIDGET --> LIST[core/modules/widgets/list.js]
    WIDGET --> EDIT[core/modules/widgets/edit.js]
    WIDGET --> LINK[core/modules/widgets/link.js]
    WIDGET --> TRANSCLUDE[core/modules/widgets/transclude.js]
    
    %% Parser System - Content Processing
    PARSER --> WIKIPARSER[core/modules/parsers/wikiparser/]
    PARSER --> TEXTPARSER[core/modules/parsers/textparser.js]
    
    %% Data Flow
    WIKIPARSER --> WIDGET
    TEXTPARSER --> WIDGET
    FILTER --> LIST
    MACRO --> TRANSCLUDE
    
    %% Storage Layer
    WIKI --> SAVER[core/modules/savers/]
    SAVER --> BOOT
    
    %% Editions System
    EDITIONS[editions/] --> BOOT
    EDITIONS --> WIKI
    
    %% Key Relationships
    TIDDLER -.->|stores| WIKI
    WIKI -.->|queries| FILTER
    WIDGET -.->|renders| LIST
    PARSER -.->|parses| WIKIPARSER
    
    %% Styling
    classDef core fill:#e1f5fe
    classDef ui fill:#f3e5f5
    classDef data fill:#e8f5e8
    classDef config fill:#fff3e0
    
    class BOOT,WIKI,TIDDLER core
    class WIDGET,LIST,EDIT,LINK,TRANSCLUDE ui
    class PARSER,FILTER,MACRO,SAVER data
    class EDITIONS config
```

## Graph Explanation

### Core Dependencies (Solid Lines)
- **Boot System** → **Wiki Engine**: Boot initializes the wiki
- **Wiki Engine** → **All Systems**: Wiki coordinates all functionality
- **Widget System** → **UI Components**: Widgets render interface elements
- **Parser System** → **Widget System**: Parsed content feeds into widgets

### Data Relationships (Dotted Lines)
- **Tiddler** → **Wiki**: Tiddlers are stored in the wiki
- **Wiki** → **Filters**: Wiki provides data for filtering
- **Widgets** → **List**: Widgets render list components
- **Parser** → **WikiParser**: Parsers process content

### Key Architectural Principles

1. **Centralized Wiki Engine**: All data flows through the wiki
2. **Widget Hierarchy**: UI components are organized in a tree structure
3. **Parser Pipeline**: Content is parsed before rendering
4. **Module System**: Functionality is organized by type
5. **Edition Configuration**: Different deployments use different module sets

### Minimal Spanning Tree Properties

- **Connected**: All components are reachable from the boot system
- **Acyclic**: No circular dependencies in the core architecture
- **Minimal**: Only essential relationships shown
- **Hierarchical**: Clear parent-child relationships

This graph shows the minimal set of connections needed to understand how TiddlyWiki's components interact, focusing on the most critical dependencies that define the system's architecture. 