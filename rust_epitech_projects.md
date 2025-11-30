}

impl DhtNode {
    pub async fn find_peers(&mut self, info_hash: [u8; 20]) -> Vec<PeerInfo> {
        // Recherche Kademlia
        // 1. Trouver K nodes les plus proches de info_hash
        // 2. Leur demander get_peers
        // 3. Itérer jusqu'à convergence
    }
}
```

**Disk I/O optimisé** :

```rust
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub struct DiskManager {
    files: Vec<File>,
    piece_length: usize,
    cache: LruCache<u32, Vec<u8>>,
}

impl DiskManager {
    pub async fn write_piece(&mut self, index: u32, data: Vec<u8>) 
        -> Result<()> 
    {
        // Calculer offset dans fichier(s)
        let offset = index as u64 * self.piece_length as u64;
        
        // Écrire de manière asynchrone
        // Gérer cas multi-fichiers (pièce à cheval)
        
        // Mettre en cache
        self.cache.put(index, data.clone());
        
        Ok(())
    }
    
    pub async fn read_piece(&mut self, index: u32) -> Result<Vec<u8>> {
        // Vérifier cache d'abord
        if let Some(data) = self.cache.get(&index) {
            return Ok(data.clone());
        }
        
        // Lire depuis disque
        let offset = index as u64 * self.piece_length as u64;
        // ... lecture async
    }
}
```

**Interface utilisateur TUI** :

```rust
use tui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph},
    Terminal,
};

pub struct TorrentUI {
    torrents: Vec<TorrentState>,
}

pub struct TorrentState {
    name: String,
    progress: f64,
    download_rate: f64,
    upload_rate: f64,
    peers: usize,
    seeds: usize,
    eta: Duration,
}

impl TorrentUI {
    pub fn render(&mut self, terminal: &mut Terminal<CrosstermBackend<Stdout>>) {
        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),  // Header
                    Constraint::Min(0),     // Torrent list
                    Constraint::Length(3),  // Status bar
                ])
                .split(f.size());
            
            // Liste des torrents avec progress bars
            let items: Vec<ListItem> = self.torrents
                .iter()
                .map(|t| {
                    let content = format!(
                        "{} - {:.1}% | ↓{}/s ↑{}/s | Peers: {} | ETA: {}",
                        t.name,
                        t.progress * 100.0,
                        format_size(t.download_rate),
                        format_size(t.upload_rate),
                        t.peers,
                        format_duration(t.eta),
                    );
                    ListItem::new(content)
                })
                .collect();
            
            let list = List::new(items)
                .block(Block::default().borders(Borders::ALL).title("Torrents"));
            f.render_widget(list, chunks[1]);
            
            // Gauge de progression
            let progress = self.torrents[0].progress;
            let gauge = Gauge::default()
                .block(Block::default().borders(Borders::ALL))
                .gauge_style(Style::default().fg(Color::Green))
                .percent((progress * 100.0) as u16);
            f.render_widget(gauge, chunks[2]);
        }).unwrap();
    }
}
```

**Configuration** :

```toml
# config.toml
[client]
peer_id_prefix = "-MY0001-"
port = 6881
max_peers_per_torrent = 50
max_upload_rate = 1048576  # 1 MB/s
max_download_rate = 0      # Unlimited

[disk]
download_dir = "./downloads"
cache_size = 67108864      # 64 MB
allocation_mode = "sparse" # or "full"

[dht]
enabled = true
bootstrap_nodes = [
    "router.bittorrent.com:6881",
    "dht.transmissionbt.com:6881"
]
```

**Features avancées** :

1. **Encryption** (MSE/PE) :
```rust
// Obfuscation du protocole pour éviter throttling ISP
pub struct EncryptedStream {
    stream: TcpStream,
    cipher: Rc4,
}
```

2. **UPnP/NAT-PMP** :
```rust
// Port forwarding automatique
pub async fn setup_port_forwarding(port: u16) -> Result<()> {
    // Utiliser igd crate
}
```

3. **Web seed** (BEP 19) :
```rust
// Télécharger depuis serveur HTTP en complément
pub async fn download_from_webseed(url: &str, piece: u32) -> Result<Vec<u8>>;
```

4. **Magnet links** :
```rust
// Supporter magnet:?xt=urn:btih:...
pub fn parse_magnet(uri: &str) -> Result<MagnetInfo> {
    // Parser et récupérer metadata via DHT/peers
}
```

**Tests et validation** :

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_bencode_parse() {
        let data = b"d8:announce...e";
        let torrent = parse_torrent(data).unwrap();
        assert_eq!(torrent.info.piece_length, 262144);
    }
    
    #[test]
    fn test_piece_hash_validation() {
        let piece = create_test_piece();
        assert!(piece.verify_hash());
    }
    
    #[tokio::test]
    async fn test_peer_connection() {
        // Test handshake
        // Test message exchange
    }
}
```

**Livrables** :
- Client BitTorrent fonctionnel (CLI + TUI)
- Support protocole complet
- Documentation protocole
- Tests d'interopérabilité avec clients existants
- Benchmarks de performance
- Guide d'utilisation

---

## Projet 10 : my_db - Base de données relationnelle (Semaines 18-19)
**Durée** : 14 jours | **Difficulté** : ⭐⭐⭐⭐⭐

### Objectifs pédagogiques
- Architecture de SGBD
- Structures de données persistantes
- Transactions ACID
- Query parsing et optimisation

### Description
Créer une base de données relationnelle from scratch avec SQL, transactions, indexes, et optimisations de requêtes.

### Architecture globale

```
my_db/
├── src/
│   ├── storage/
│   │   ├── page.rs          // Page-based storage
│   │   ├── heap_file.rs     // Heap file manager
│   │   ├── buffer_pool.rs   // Buffer pool manager
│   │   └── wal.rs           // Write-Ahead Log
│   ├── index/
│   │   ├── btree.rs         // B+ tree index
│   │   └── hash.rs          // Hash index
│   ├── catalog/
│   │   ├── schema.rs        // Table schemas
│   │   └── metadata.rs      // System catalog
│   ├── sql/
│   │   ├── parser.rs        // SQL parser
│   │   ├── lexer.rs         // SQL lexer
│   │   └── ast.rs           // Abstract syntax tree
│   ├── executor/
│   │   ├── planner.rs       // Query planner
│   │   ├── optimizer.rs     // Query optimizer
│   │   └── operators.rs     // Execution operators
│   ├── transaction/
│   │   ├── manager.rs       // Transaction manager
│   │   ├── lock.rs          // Lock manager
│   │   └── mvcc.rs          // MVCC implementation
│   └── server/
│       ├── protocol.rs      // Wire protocol
│       └── session.rs       // Client session
```

### Storage Layer

**Page-based storage** :

```rust
pub const PAGE_SIZE: usize = 8192; // 8KB pages

#[repr(C)]
pub struct Page {
    header: PageHeader,
    data: [u8; PAGE_SIZE - std::mem::size_of::<PageHeader>()],
}

#[repr(C)]
pub struct PageHeader {
    page_id: u32,
    page_type: PageType,
    free_space: u16,
    num_tuples: u16,
    next_page: u32,
    checksum: u32,
}

pub enum PageType {
    Data,
    Index,
    Overflow,
    FreeSpace,
}

impl Page {
    pub fn new(page_id: u32, page_type: PageType) -> Self {
        // Initialiser page
    }
    
    pub fn insert_tuple(&mut self, tuple: &[u8]) -> Result<SlotId> {
        // Vérifier espace disponible
        if self.free_space() < tuple.len() {
            return Err(Error::PageFull);
        }
        
        // Insérer tuple
        let slot_id = self.allocate_slot();
        self.write_tuple(slot_id, tuple);
        self.header.num_tuples += 1;
        
        Ok(slot_id)
    }
    
    pub fn get_tuple(&self, slot_id: SlotId) -> Result<&[u8]> {
        // Récupérer tuple par slot
    }
    
    pub fn delete_tuple(&mut self, slot_id: SlotId) -> Result<()> {
        // Marquer slot comme libre
    }
}

pub type SlotId = u16;
pub type PageId = u32;
pub type TupleId = (PageId, SlotId);
```

**Buffer Pool Manager** :

```rust
pub struct BufferPool {
    frames: Vec<Frame>,
    page_table: HashMap<PageId, FrameId>,
    replacer: LruReplacer,
    disk_manager: DiskManager,
}

pub struct Frame {
    page: Page,
    pin_count: AtomicU32,
    dirty: AtomicBool,
}

impl BufferPool {
    pub fn fetch_page(&mut self, page_id: PageId) -> Result<&mut Page> {
        // Vérifier si déjà en mémoire
        if let Some(&frame_id) = self.page_table.get(&page_id) {
            self.frames[frame_id].pin_count.fetch_add(1, Ordering::SeqCst);
            return Ok(&mut self.frames[frame_id].page);
        }
        
        // Trouver frame victime
        let frame_id = self.replacer.victim()?;
        
        // Flush si dirty
        if self.frames[frame_id].dirty.load(Ordering::SeqCst) {
            self.flush_page(frame_id)?;
        }
        
        // Charger nouvelle page
        self.disk_manager.read_page(page_id, &mut self.frames[frame_id].page)?;
        self.page_table.insert(page_id, frame_id);
        self.frames[frame_id].pin_count.store(1, Ordering::SeqCst);
        
        Ok(&mut self.frames[frame_id].page)
    }
    
    pub fn unpin_page(&mut self, page_id: PageId, is_dirty: bool) {
        if let Some(&frame_id) = self.page_table.get(&page_id) {
            self.frames[frame_id].pin_count.fetch_sub(1, Ordering::SeqCst);
            if is_dirty {
                self.frames[frame_id].dirty.store(true, Ordering::SeqCst);
            }
        }
    }
}

// LRU Replacer
pub struct LruReplacer {
    lru_list: LinkedList<FrameId>,
    map: HashMap<FrameId, NodePtr>,
}

impl LruReplacer {
    pub fn victim(&mut self) -> Result<FrameId> {
        // Retourner LRU frame non-pinned
    }
    
    pub fn pin(&mut self, frame_id: FrameId) {
        // Retirer de LRU list
    }
    
    pub fn unpin(&mut self, frame_id: FrameId) {
        // Ajouter à LRU list
    }
}
```

### Index Layer - B+ Tree

```rust
pub struct BPlusTree<K: Ord, V> {
    root: NodeId,
    buffer_pool: Arc<Mutex<BufferPool>>,
    order: usize,  // Max children per node
}

pub enum Node<K, V> {
    Internal(InternalNode<K>),
    Leaf(LeafNode<K, V>),
}

pub struct InternalNode<K> {
    keys: Vec<K>,
    children: Vec<NodeId>,
}

pub struct LeafNode<K, V> {
    keys: Vec<K>,
    values: Vec<V>,
    next_leaf: Option<NodeId>,
}

impl<K: Ord + Clone, V: Clone> BPlusTree<K, V> {
    pub fn insert(&mut self, key: K, value: V) -> Result<()> {
        // 1. Trouver leaf node appropriée
        let leaf_id = self.find_leaf(&key);
        let mut leaf = self.get_leaf_mut(leaf_id)?;
        
        // 2. Insérer dans leaf
        let insert_pos = leaf.keys.binary_search(&key)
            .unwrap_or_else(|pos| pos);
        leaf.keys.insert(insert_pos, key.clone());
        leaf.values.insert(insert_pos, value);
        
        // 3. Split si nécessaire
        if leaf.keys.len() > self.order {
            self.split_leaf(leaf_id)?;
        }
        
        Ok(())
    }
    
    pub fn search(&self, key: &K) -> Result<Option<V>> {
        let leaf_id = self.find_leaf(key);
        let leaf = self.get_leaf(leaf_id)?;
        
        match leaf.keys.binary_search(key) {
            Ok(pos) => Ok(Some(leaf.values[pos].clone())),
            Err(_) => Ok(None),
        }
    }
    
    pub fn range_scan(&self, start: &K, end: &K) -> Result<Vec<V>> {
        // Scanner plusieurs leaf nodes via next_leaf pointer
        let mut results = Vec::new();
        let mut current_leaf = self.find_leaf(start);
        
        loop {
            let leaf = self.get_leaf(current_leaf)?;
            
            for (k, v) in leaf.keys.iter().zip(&leaf.values) {
                if k >= start && k <= end {
                    results.push(v.clone());
                }
                if k > end {
                    return Ok(results);
                }
            }
            
            match leaf.next_leaf {
                Some(next) => current_leaf = next,
                None => break,
            }
        }
        
        Ok(results)
    }
    
    fn split_leaf(&mut self, leaf_id: NodeId) -> Result<()> {
        // Split leaf en deux
        // Update parent internal node
        // Propagate split si parent full
    }
}
```

### SQL Parser

```rust
use nom::{
    IResult,
    bytes::complete::{tag, tag_no_case, take_while1},
    character::complete::{alpha1, digit1, multispace0},
    combinator::map,
    multi::separated_list1,
    sequence::{delimited, preceded, tuple},
};

#[derive(Debug, Clone)]
pub enum Statement {
    Select(SelectStmt),
    Insert(InsertStmt),
    Update(UpdateStmt),
    Delete(DeleteStmt),
    CreateTable(CreateTableStmt),
    CreateIndex(CreateIndexStmt),
    BeginTransaction,
    Commit,
    Rollback,
}

#[derive(Debug, Clone)]
pub struct SelectStmt {
    pub columns: Vec<ColumnRef>,
    pub from: Vec<TableRef>,
    pub where_clause: Option<Expr>,
    pub group_by: Vec<Expr>,
    pub having: Option<Expr>,
    pub order_by: Vec<OrderByExpr>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct CreateTableStmt {
    pub name: String,
    pub columns: Vec<ColumnDef>,
    pub primary_key: Option<Vec<String>>,
    pub foreign_keys: Vec<ForeignKeyDef>,
}

#[derive(Debug, Clone)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub default: Option<Literal>,
    pub unique: bool,
}

#[derive(Debug, Clone)]
pub enum DataType {
    Integer,
    BigInt,
    Real,
    Double,
    Varchar(usize),
    Text,
    Boolean,
    Date,
    Timestamp,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Literal(Literal),
    ColumnRef(ColumnRef),
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOperator,
        right: Box<Expr>,
    },
    UnaryOp {
        op: UnaryOperator,
        expr: Box<Expr>,
    },
    Function {
        name: String,
        args: Vec<Expr>,
    },
    Subquery(Box<SelectStmt>),
}

// Parser SELECT
fn parse_select(input: &str) -> IResult<&str, SelectStmt> {
    let (input, _) = tag_no_case("SELECT")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, columns) = parse_column_list(input)?;
    let (input, _) = multispace1(input)?;
    let (input, _) = tag_no_case("FROM")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, from) = parse_table_list(input)?;
    
    // WHERE clause optionnelle
    let (input, where_clause) = opt(preceded(
        tuple((multispace1, tag_no_case("WHERE"), multispace1)),
        parse_expr
    ))(input)?;
    
    // ... autres clauses
    
    Ok((input, SelectStmt {
        columns,
        from,
        where_clause,
        // ...
    }))
}
```

### Query Executor

**Logical plan** :

```rust
#[derive(Debug)]
pub enum LogicalPlan {
    Scan {
        table: String,
        filter: Option<Expr>,
    },
    IndexScan {
        table: String,
        index: String,
        key_range: (Bound<Value>, Bound<Value>),
    },
    Join {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        join_type: JoinType,
        condition: Expr,
    },
    Aggregate {
        input: Box<LogicalPlan>,
        group_by: Vec<Expr>,
        aggregates: Vec<AggregateExpr>,
    },
    Sort {
        input: Box<LogicalPlan>,
        order_by: Vec<OrderByExpr>,
    },
    Limit {
        input: Box<LogicalPlan>,
        limit: usize,
        offset: usize,
    },
}

pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
}
```

**Physical plan** :

```rust
pub trait PhysicalOperator {
    fn open(&mut self) -> Result<()>;
    fn next(&mut self) -> Result<Option<Tuple>>;
    fn close(&mut self) -> Result<()>;
}

// Sequential scan
pub struct SeqScan {
    table: Table,
    filter: Option<Expr>,
    current_page: PageId,
    current_slot: SlotId,
}

impl PhysicalOperator for SeqScan {
    fn next(&mut self) -> Result<Option<Tuple>> {
        // Itérer sur pages et tuples
        // Appliquer filtre si présent
    }
}

// Index scan
pub struct IndexScan {
    index: BPlusTree<Value, TupleId>,
    key_range: (Bound<Value>, Bound<Value>),
    iterator: RangeIterator,
}

// Nested loop join
pub struct NestedLoopJoin {
    left: Box<dyn PhysicalOperator>,
    right: Box<dyn PhysicalOperator>,
    condition: Expr,
    current_left: Option<Tuple>,
}

impl PhysicalOperator for NestedLoopJoin {
    fn next(&mut self) -> Result<Option<Tuple>> {
        loop {
            if self.current_left.is_none() {
                self.current_left = self.left.next()?;
                if self.current_left.is_none() {
                    return Ok(None);
                }
            }
            
            if let Some(right_tuple) = self.right.next()? {
                let combined = self.combine_tuples(
                    self.current_left.as_ref().unwrap(),
                    &right_tuple
                );
                
                if self.evaluate_condition(&combined)? {
                    return Ok(Some(combined));
                }
            } else {
                self.right.open()?; // Reset right
                self.current_left = None;
            }
        }
    }
}

// Hash join (plus efficace)
pub struct HashJoin {
    left: Box<dyn PhysicalOperator>,
    right: Box<dyn PhysicalOperator>,
    join_keys: Vec<usize>,
    hash_table: HashMap<Vec<Value>, Vec<Tuple>>,
}
```

### Transaction Management

**MVCC (Multi-Version Concurrency Control)** :

```rust
pub struct TransactionManager {
    next_txn_id: AtomicU64,
    active_transactions: RwLock<HashSet<TxnId>>,
    lock_manager: LockManager,
}

pub type TxnId = u64;

pub struct Transaction {
    txn_id: TxnId,
    start_timestamp: u64,
    isolation_level: IsolationLevel,
    undo_log: Vec<UndoRecord>,
}

pub enum IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    Serializable,
}

// Chaque tuple a version info
pub struct TupleHeader {
    xmin: TxnId,  // Transaction qui a créé
    xmax: TxnId,  // Transaction qui a supprimé (0 si actif)
    cmin: u32,    // Command ID dans transaction
    cmax: u32,
}

impl Transaction {
    pub fn is_visible(&self, header: &TupleHeader) -> bool {
        // MVCC visibility rules
        // Un tuple est visible si:
        // 1. xmin committed avant notre start_timestamp
        // 2. xmax = 0 OU xmax committed après notre start
        
        let xmin_committed = self.is_committed(header.xmin);
        let xmax_active = header.xmax == 0 || !self.is_committed(header.xmax);
        
        xmin_committed && xmax_active
    }
}
```

**Write-Ahead Log** :

```rust
pub struct WAL {
    log_file: File,
    buffer: Vec<LogRecord>,
    lsn: AtomicU64,  // Log Sequence Number
}

pub enum LogRecord {
    Insert {
        txn_id: TxnId,
        table_id: TableId,
        tuple_id: TupleId,
        tuple_data: Vec<u8>,
    },
    Delete {
        txn_id: TxnId,
        table_id: TableId,
        tuple_id: TupleId,
    },
    Update {
        txn_id: TxnId,
        table_id: TableId,
        tuple_id: TupleId,
        old_data: Vec<u8>,
        new_data: Vec<u8>,
    },
    Commit {
        txn_id: TxnId,
    },
    Abort {
        txn_id: TxnId,
    },
}

impl WAL {
    pub fn append(&mut self, record: LogRecord) -> Result<LSN> {
        // Serializer record
        let bytes = bincode::serialize(&record)?;
        
        // Écrire dans buffer
        self.buffer.extend_from_slice(&bytes);
        
        // Flush si buffer plein
        if self.buffer.len() > WAL_BUFFER_SIZE {
            self.flush()?;
        }
        
        let lsn = self.lsn.fetch_add(1, Ordering::SeqCst);
        Ok(lsn)
    }
    
    pub fn flush(&mut self) -> Result<()> {
        self.log_file.write_all(&self.buffer)?;
        self.log_file.sync_all()?;
        self.buffer.clear();
        Ok(())
    }
    
    pub fn recover(&mut self) -> Result<()> {
        // ARIES recovery algorithm
        // 1. Analysis phase
        // 2. Redo phase
        // 3. Undo phase
    }
}
```

**Lock Manager** :

```rust
pub struct LockManager {
    locks: RwLock<HashMap<ResourceId, LockQueue>>,
}

pub enum LockMode {
    Shared,
    Exclusive,
    IntentionShared,
    IntentionExclusive,
}

pub struct LockQueue {
    holders: Vec<(TxnId, LockMode)>,
    waiters: VecDeque<LockRequest>,
}

impl LockManager {
    pub fn acquire_lock(
        &self,
        txn_id: TxnId,
        resource: ResourceId,
        mode: LockMode,
    ) -> Result<()> {
        let mut locks = self.locks.write();
        let queue = locks.entry(resource).or_insert_with(LockQueue::new);
        
        // Vérifier compatibilité
        if queue.is_compatible(mode) {
            queue.holders.push((txn_id, mode));
            Ok(())
        } else {
            // Attendre ou deadlock detection
            queue.waiters.push_back(LockRequest { txn_id, mode });
            Err(Error::WouldBlock)
        }
    }
    
    pub fn detect_deadlock(&self) -> Option<TxnId> {
        // Wait-for graph
        // Cycle detection
    }
}
```

### Query Optimizer

```rust
pub struct Optimizer {
    catalog: Catalog,
    statistics: Statistics,
}

impl Optimizer {
    pub fn optimize(&self, logical_plan: LogicalPlan) -> PhysicalPlan {
        let mut plan = logical_plan;
        
        // Rule-based optimization
        plan = self.push_down_predicates(plan);
        plan = self.push_down_projections(plan);
        plan = self.merge_predicates(plan);
        
        // Cost-based optimization
        let physical_plans = self.generate_physical_plans(plan);
        let best_plan = physical_plans
            .into_iter()
            .min_by_key(|p| self.estimate_cost(p))
            .unwrap();
        
        best_plan
    }
    
    fn estimate_cost(&self, plan: &PhysicalPlan) -> Cost {
        match plan {
            PhysicalPlan::SeqScan { table, .. } => {
                let stats = self.statistics.get_table_stats(table);
                Cost {
                    cpu: stats.num_tuples,
                    io: stats.num_pages,
                }
            }
            PhysicalPlan::IndexScan { selectivity, .. } => {
                // Estimer coût basé sur sélectivité
            }
            PhysicalPlan::HashJoin { left, right, .. } => {
                let left_cost = self.estimate_cost(left);
                let right_cost = self.estimate_cost(right);
                // Build hash table + probe
                left_cost + right_cost + Cost { cpu: 1000, io: 0 }
            }
            // ...
        }
    }
}
```

### Network Protocol

```rust
// PostgreSQL wire protocol compatible

pub struct Server {
    listener: TcpListener,
    sessions: Vec<Session>,
}

pub struct Session {
    stream: TcpStream,
    transaction: Option<Transaction>,
    prepared_statements: HashMap<String, PreparedStatement>,
}

#[derive(Debug)]
pub enum Message {
    Query(String),
    Parse {
        name: String,
        query: String,
        param_types: Vec<DataType>,
    },
    Bind {
        statement_name: String,
        params: Vec<Value>,
    },
    Execute {
        portal_name: String,
        max_rows: i32,
    },
    Sync,
    Terminate,
}

impl Session {
    pub async fn handle_message(&mut self, msg: Message) -> Result<Response> {
        match msg {
            Message::Query(sql) => {
                let stmt = parse_sql(&sql)?;
                let plan = plan_query(stmt)?;
                let results = execute_plan(plan)?;
                Ok(Response::RowDescription(results))
            }
            Message::Parse { name, query, .. } => {
                let stmt = parse_sql(&query)?;
                let prepared = PreparedStatement::new(stmt);
                self.prepared_statements.insert(name, prepared);
                Ok(Response::ParseComplete)
            }
            // ...
        }
    }
}
```

**Livrables** :
- SGBD fonctionnel
- Support SQL basique (SELECT, INSERT, UPDATE, DELETE, CREATE TABLE)
- Transactions ACID
- Indexes B+ tree
- Documentation architecture
- Benchmarks TPC-C/TPC-H
- Guide d'utilisation

---

# 🎯 MOIS 6 : Projet final et portfolio

## Projet 11 : my_kubernetes - Container orchestrator (Semaines 20-22)
**Durée** : 21 jours | **Difficulté** : ⭐⭐⭐⭐⭐

### Objectifs pédagogiques
- Systèmes distribués
- Consensus et coordination# Programme de maîtrise Rust
## Projets Epitech-style - 6 mois intensifs

---

# 🎯 MOIS 1 : Fondamentaux et manipulation de données

## Projet 1 : my_sqrt - Calculateur de racine carrée (Semaine 1)
**Durée** : 7 jours | **Difficulté** : ⭐☆☆☆☆

### Objectifs pédagogiques
- Maîtriser les boucles et conditions
- Comprendre la précision des flottants
- Gestion des erreurs basique
- Tests unitaires simples

### Description
Implémenter un algorithme de calcul de racine carrée sans utiliser les fonctions de la bibliothèque standard. Support des nombres négatifs avec nombres complexes.

### Fonctionnalités requises
```rust
// API attendue
fn my_sqrt(n: f64) -> Result<f64, String>
fn my_sqrt_complex(n: f64) -> (f64, f64) // (partie réelle, imaginaire)
fn my_sqrt_integer(n: u64) -> u64 // racine entière
```

**Features à implémenter** :
1. Méthode de Newton-Raphson
2. Méthode babylonienne
3. Comparaison de performance entre méthodes
4. Gestion précision (epsilon configurable)
5. Support nombres négatifs (i²=-1)
6. CLI avec choix de l'algorithme

**Contraintes techniques** :
- Pas d'utilisation de `sqrt()` de std
- Précision minimum : 10⁻⁶
- Temps de calcul < 1ms pour nombres < 10⁶
- Tests avec property-based testing

**Livrables** :
- Code source avec documentation
- Rapport comparatif des algorithmes
- Benchmarks de performance
- Suite de tests (>80% coverage)

---

## Projet 2 : my_ls - Clone de la commande ls (Semaines 2-3)
**Durée** : 14 jours | **Difficulté** : ⭐⭐☆☆☆

### Objectifs pédagogiques
- Manipulation du système de fichiers
- Parsing d'arguments CLI
- Formatage de sortie complexe
- Gestion des permissions et métadonnées

### Description
Recoder la commande Unix `ls` avec toutes ses options principales. Affichage en colonnes, tri, filtrage, et couleurs.

### Fonctionnalités requises

**Options obligatoires** :
```bash
my_ls [OPTIONS] [PATH...]

-l, --long          Format long avec détails
-a, --all          Afficher fichiers cachés
-R, --recursive    Lister récursivement
-t                 Trier par date de modification
-S                 Trier par taille
-r, --reverse      Inverser l'ordre
-h, --human        Tailles lisibles (KB, MB, GB)
--color[=WHEN]     Colorer la sortie
-1                 Un fichier par ligne
```

**Architecture attendue** :
```
my_ls/
├── src/
│   ├── main.rs
│   ├── cli.rs          // Parsing arguments avec clap
│   ├── file_info.rs    // Struct pour métadonnées
│   ├── formatter.rs    // Formatage output
│   ├── sorter.rs       // Logique de tri
│   └── colors.rs       // Gestion couleurs
├── tests/
│   ├── integration_tests.rs
│   └── fixtures/       // Fichiers de test
└── benches/
    └── ls_bench.rs
```

**Détails d'implémentation** :

1. **Struct FileInfo** :
```rust
struct FileInfo {
    name: String,
    path: PathBuf,
    size: u64,
    modified: SystemTime,
    permissions: Permissions,
    is_dir: bool,
    is_symlink: bool,
    owner: String,
    group: String,
}
```

2. **Formatage en colonnes** :
- Calculer largeur optimale selon terminal
- Aligner les colonnes proprement
- Gérer les noms longs

3. **Couleurs** :
- Bleu pour dossiers
- Vert pour exécutables
- Cyan pour symlinks
- Rouge pour archives
- Support NO_COLOR env var

**Bonus** :
- Support des glob patterns (`*.rs`)
- Option `-i` pour inodes
- Format tree view
- Statistiques (nombre fichiers, taille totale)

**Contraintes** :
- Performance : gérer 10000+ fichiers instantanément
- Compatibilité POSIX
- Tests sur Linux et macOS
- Gestion erreurs (permissions, fichiers supprimés)

---

## Projet 3 : my_sokoban - Jeu de puzzle dans le terminal (Semaine 4)
**Durée** : 7 jours | **Difficulté** : ⭐⭐☆☆☆

### Objectifs pédagogiques
- Boucle de jeu et gestion d'état
- Input/output terminal
- Algorithmes de pathfinding
- Structures de données 2D

### Description
Implémenter le jeu classique Sokoban dans le terminal avec interface TUI, éditeur de niveaux, et système de sauvegarde.

### Fonctionnalités requises

**Gameplay** :
- Grille de jeu avec murs, caisses, cibles
- Déplacement joueur (↑↓←→)
- Pousser les caisses (pas tirer)
- Détection victoire (toutes caisses sur cibles)
- Undo/redo des mouvements
- Reset du niveau

**Interface TUI** :
```
┌─ Sokoban - Niveau 1 ──────────┐
│                                │
│   ████████████                 │
│   █  . $     █                 │
│   █    @     █                 │
│   █  $ . $  █                  │
│   █         █                   │
│   ████████████                 │
│                                │
│ Mouvements: 42 | Undo: U       │
│ Reset: R | Quit: Q             │
└────────────────────────────────┘

Légende:
@ = Joueur    $ = Caisse    . = Cible
* = Caisse sur cible    # = Mur
```

**Architecture** :
```rust
struct Game {
    grid: Grid,
    player: Position,
    boxes: Vec<Position>,
    targets: Vec<Position>,
    moves: Vec<Move>,
    level: usize,
}

struct Grid {
    width: usize,
    height: usize,
    cells: Vec<Vec<Cell>>,
}

enum Cell {
    Empty,
    Wall,
    Target,
}

enum Direction {
    Up, Down, Left, Right
}
```

**Features avancées** :
1. **Éditeur de niveaux** :
   - Mode édition avec menu
   - Placement interactif d'éléments
   - Validation du niveau (solvable)
   - Sauvegarde format texte

2. **Solver automatique** :
   - A* pour trouver solution optimale
   - Afficher la solution pas à pas
   - Détection deadlocks

3. **Système de niveaux** :
   - 20+ niveaux de difficulté croissante
   - Format de fichier personnalisé
   - Progression sauvegardée

4. **Statistiques** :
   - Meilleur score par niveau
   - Temps de résolution
   - Leaderboard local

**Contraintes techniques** :
- Utiliser `crossterm` pour TUI
- 60 FPS minimum
- Support redimensionnement terminal
- Pas de flickering

**Bonus** :
- Mode multijoueur local (tour par tour)
- Thèmes de couleurs
- Animations de mouvement
- Sound effects (terminal beep)

---

# 🎯 MOIS 2 : Structures de données et algorithmes

## Projet 4 : my_hash_map - Table de hachage from scratch (Semaines 5-6)
**Durée** : 14 jours | **Difficulté** : ⭐⭐⭐☆☆

### Objectifs pédagogiques
- Implémenter une structure de données complexe
- Comprendre le hashing et collisions
- Gérer la mémoire manuellement
- Optimisation de performance

### Description
Créer une HashMap complète sans utiliser std::collections::HashMap, avec gestion des collisions, redimensionnement automatique, et API idiomatique Rust.

### Spécifications détaillées

**API publique** :
```rust
pub struct MyHashMap<K, V> {
    // Implémentation cachée
}

impl<K, V> MyHashMap<K, V> 
where 
    K: Hash + Eq,
{
    pub fn new() -> Self;
    pub fn with_capacity(capacity: usize) -> Self;
    pub fn insert(&mut self, key: K, value: V) -> Option<V>;
    pub fn get(&self, key: &K) -> Option<&V>;
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V>;
    pub fn remove(&mut self, key: &K) -> Option<V>;
    pub fn contains_key(&self, key: &K) -> bool;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn clear(&mut self);
    pub fn capacity(&self) -> usize;
    pub fn reserve(&mut self, additional: usize);
    pub fn shrink_to_fit(&mut self);
    
    // Iterators
    pub fn iter(&self) -> Iter<K, V>;
    pub fn iter_mut(&mut self) -> IterMut<K, V>;
    pub fn keys(&self) -> Keys<K, V>;
    pub fn values(&self) -> Values<K, V>;
    pub fn values_mut(&mut self) -> ValuesMut<K, V>;
    
    // Entry API
    pub fn entry(&mut self, key: K) -> Entry<K, V>;
}
```

**Stratégies de collision à implémenter** :

1. **Chaînage séparé** (Separate Chaining) :
```rust
struct Node<K, V> {
    key: K,
    value: V,
    next: Option<Box<Node<K, V>>>,
}

struct MyHashMap<K, V> {
    buckets: Vec<Option<Box<Node<K, V>>>>,
    size: usize,
    capacity: usize,
}
```

2. **Adressage ouvert** (Open Addressing) :
- Linear probing
- Quadratic probing
- Double hashing

**Fonctions de hachage** :
- FNV-1a hash
- MurmurHash3
- SipHash (comparaison)
- Custom hash pour performance

**Métriques à tracker** :
```rust
pub struct HashMapStats {
    pub collisions: usize,
    pub max_probe_length: usize,
    pub load_factor: f64,
    pub resize_count: usize,
    pub total_lookups: usize,
}
```

**Features avancées** :

1. **Redimensionnement intelligent** :
   - Load factor configurable (défaut 0.75)
   - Doubler la capacité au besoin
   - Rehashing progressif pour éviter lag

2. **Entry API** :
```rust
let count = map.entry("hello")
    .and_modify(|e| *e += 1)
    .or_insert(0);
```

3. **Iterators personnalisés** :
   - Implémenter tous les traits Iterator
   - Support for_each optimisé
   - Parallel iterator (rayon)

4. **Optimisations** :
   - SIMD pour hashing rapide
   - Cache-friendly memory layout
   - Small string optimization

**Benchmarks requis** :
- Comparer avec std::collections::HashMap
- Tester avec 1M+ éléments
- Mesurer :
  - Temps d'insertion
  - Temps de lookup
  - Utilisation mémoire
  - Performance collisions

**Tests exhaustifs** :
- Property-based testing
- Fuzzing
- Tests de stress (millions d'opérations)
- Tests de thread safety (si concurrent)

**Livrables** :
- Bibliothèque my_hashmap
- Documentation complète
- Rapport de performance
- Visualisations des collisions
- Article de blog expliquant l'implémentation

---

## Projet 5 : my_grep - Clone de grep avec regex (Semaines 7-8)
**Durée** : 14 jours | **Difficulté** : ⭐⭐⭐☆☆

### Objectifs pédagogiques
- Parsing et regex
- Traitement de fichiers volumineux
- Optimisation I/O
- Parallélisme

### Description
Recoder grep avec support regex, recherche multi-fichiers, et performance optimale pour gérer des Go de données.

### Fonctionnalités complètes

**Options CLI** :
```bash
my_grep [OPTIONS] PATTERN [PATH...]

Options de base:
  -i, --ignore-case      Ignorer la casse
  -v, --invert-match     Inverser le match
  -c, --count           Compter les matches
  -n, --line-number     Afficher numéros de ligne
  -H, --with-filename   Afficher nom de fichier
  -h, --no-filename     Cacher nom de fichier

Options avancées:
  -r, --recursive       Recherche récursive
  -E, --extended-regex  Regex étendue (défaut)
  -F, --fixed-strings   String littérale (pas regex)
  -w, --word-regexp     Match mots entiers
  -x, --line-regexp     Match ligne entière
  -A NUM               Afficher NUM lignes après
  -B NUM               Afficher NUM lignes avant
  -C NUM               Afficher NUM lignes contexte
  --color[=WHEN]       Colorer les matches
  -l, --files-with-matches  Noms de fichiers seulement
  -L, --files-without-match Fichiers sans match
  --include=GLOB       Filtrer fichiers
  --exclude=GLOB       Exclure fichiers
  -j, --threads=NUM    Nombre de threads
```

**Architecture** :

```
my_grep/
├── src/
│   ├── main.rs
│   ├── cli.rs              // Clap config
│   ├── search/
│   │   ├── mod.rs
│   │   ├── matcher.rs      // Regex matching
│   │   ├── searcher.rs     // Core search logic
│   │   └── parallel.rs     // Parallel search
│   ├── input/
│   │   ├── mod.rs
│   │   ├── reader.rs       // Buffered file reading
│   │   └── walker.rs       // Directory traversal
│   ├── output/
│   │   ├── mod.rs
│   │   ├── printer.rs      // Format output
│   │   └── color.rs        // ANSI colors
│   └── regex/
│       ├── mod.rs
│       ├── parser.rs       // Regex parser
│       └── compiler.rs     // NFA/DFA compiler
└── tests/
```

**Implémentation regex** :

Option 1 : Utiliser `regex` crate
Option 2 : **Implémenter un moteur regex basique** (bonus)

Pour le bonus, implémenter :
```rust
// Syntaxe supportée
. : n'importe quel caractère
* : 0 ou plus
+ : 1 ou plus
? : 0 ou 1
[abc] : classe de caractères
[^abc] : classe négative
^ : début de ligne
$ : fin de ligne
| : alternation
() : groupes
\d, \w, \s : classes prédéfinies
```

**Moteur de recherche** :

```rust
pub struct Searcher {
    matcher: Box<dyn Matcher>,
    config: SearchConfig,
}

pub struct SearchConfig {
    case_insensitive: bool,
    invert_match: bool,
    line_number: bool,
    count_only: bool,
    context_before: usize,
    context_after: usize,
}

pub trait Matcher: Send + Sync {
    fn is_match(&self, text: &str) -> bool;
    fn find(&self, text: &str) -> Option<Match>;
}
```

**Optimisations critiques** :

1. **I/O optimisé** :
   - Memory-mapped files pour gros fichiers
   - Buffering intelligent
   - Lecture par chunks

2. **Parallélisme** :
   - Rayon pour multi-fichiers
   - Work-stealing
   - Chaque fichier = task indépendant

3. **SIMD** :
   - Recherche de patterns simples avec SIMD
   - Avx2/SSE pour x86
   - Fallback portable

**Tests de performance** :

Benchmarks contre :
- GNU grep
- ripgrep (référence moderne)
- ag (the silver searcher)

Scénarios :
- Recherche dans 1 fichier de 1GB
- Recherche dans 100,000 petits fichiers
- Regex complexe
- Case-insensitive sur texte Unicode

**Features bonus** :
- Support encodings (UTF-8, UTF-16, Latin-1)
- .gitignore respect
- Format JSON output
- Recherche dans fichiers compressés (.gz)

**Livrables** :
- Binary optimisé (< 5MB)
- Benchmarks détaillés
- Profiling reports
- Documentation utilisateur
- Comparaison avec concurrents

---

# 🎯 MOIS 3 : Graphique et systèmes

## Projet 6 : my_paint - Éditeur graphique bitmap (Semaines 9-10)
**Durée** : 14 jours | **Difficulté** : ⭐⭐⭐⭐☆

### Objectifs pédagogiques
- Programmation graphique 2D
- Gestion événements souris/clavier
- Algorithmes de dessin (Bresenham)
- Manipulation pixels et couleurs

### Description
Créer un éditeur d'images bitmap avec outils de dessin, filtres, calques, et support formats image standards (PNG, JPG).

### Interface et fonctionnalités

**GUI avec winit + pixels** :
```
┌─────────────────────────────────────────┐
│ File  Edit  Image  Filters  Help       │  Menu bar
├──┬──────────────────────────────────────┤
│🖌️│                                      │
│✏️│                                      │
│⬜│         CANVAS                       │  Outils
│🎨│         800x600                      │  +
│🔲│                                      │  Canvas
│⚫│                                      │
│  │                                      │
├──┴──────────────────────────────────────┤
│ Color: #FF0000  Size: 5px  Layer: 1    │  Status bar
└─────────────────────────────────────────┘
```

**Outils de dessin** :

1. **Crayon** (Pencil) :
   - Dessin libre à main levée
   - Taille configurable (1-50px)
   - Anti-aliasing

2. **Pinceau** (Brush) :
   - Différentes formes (rond, carré, étoile)
   - Opacité variable
   - Blend modes

3. **Gomme** (Eraser) :
   - Efface vers transparence
   - Taille ajustable

4. **Remplissage** (Fill) :
   - Flood fill algorithm
   - Tolérance de couleur
   - Anti-aliasing

5. **Formes** :
   - Ligne (Bresenham)
   - Rectangle (rempli/vide)
   - Cercle (Midpoint)
   - Ellipse
   - Polygone

6. **Sélection** :
   - Rectangle
   - Lasso (main levée)
   - Baguette magique (color-based)
   - Opérations : copier/coller/couper

7. **Texte** :
   - Police configurable
   - Couleur et taille
   - Rendering avec rusttype

**Structure de données** :

```rust
pub struct Canvas {
    layers: Vec<Layer>,
    active_layer: usize,
    width: u32,
    height: u32,
    history: Vec<CanvasState>,  // Undo/redo
}

pub struct Layer {
    name: String,
    pixels: Vec<Color>,
    visible: bool,
    opacity: f32,
    blend_mode: BlendMode,
}

pub struct Color {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

pub enum BlendMode {
    Normal,
    Multiply,
    Screen,
    Overlay,
    Add,
}

pub struct Tool {
    kind: ToolKind,
    size: u32,
    color: Color,
    opacity: f32,
}

pub enum ToolKind {
    Pencil,
    Brush(BrushShape),
    Eraser,
    Fill,
    Line,
    Rectangle { filled: bool },
    Circle { filled: bool },
    Select(SelectMode),
}
```

**Algorithmes de dessin** :

1. **Ligne de Bresenham** :
```rust
fn draw_line(
    canvas: &mut Canvas,
    x0: i32, y0: i32,
    x1: i32, y1: i32,
    color: Color
) {
    // Implémentation efficace
    // Seulement opérations entières
}
```

2. **Cercle de Midpoint** :
```rust
fn draw_circle(
    canvas: &mut Canvas,
    cx: i32, cy: i32,
    radius: u32,
    color: Color
) {
    // Utiliser symétrie 8-ways
}
```

3. **Flood Fill** :
```rust
// Scanline flood fill (plus efficace que récursif)
fn flood_fill(
    canvas: &mut Canvas,
    start_x: u32, start_y: u32,
    target_color: Color,
    replacement_color: Color
) {
    let mut stack = Vec::new();
    // Implémentation avec stack
}
```

**Filtres d'image** :

1. **Filtres de base** :
   - Grayscale
   - Invert colors
   - Brightness/Contrast
   - Hue/Saturation
   - Temperature (warm/cool)

2. **Convolution** :
```rust
// Matrice de convolution générique
fn apply_kernel(canvas: &Canvas, kernel: &[[f32; 3]; 3]) -> Canvas;

// Filtres prédéfinis
- Blur (Gaussian)
- Sharpen
- Edge detection (Sobel)
- Emboss
```

3. **Filtres avancés** :
   - Median filter (réduction bruit)
   - Bilateral filter
   - Sepia tone
   - Vignette

**Système de calques** :

- Ajout/suppression de calques
- Réorganisation (drag & drop)
- Fusion de calques
- Blend modes entre calques
- Masques de calque (bonus)

**Undo/Redo** :

```rust
struct History {
    states: Vec<CanvasState>,
    current: usize,
    max_size: usize,
}

impl History {
    fn push(&mut self, state: CanvasState);
    fn undo(&mut self) -> Option<&CanvasState>;
    fn redo(&mut self) -> Option<&CanvasState>;
}
```

**Import/Export** :

Formats supportés :
- PNG (avec transparence)
- JPEG
- BMP
- Format natif .mypaint (préserve calques)

Utiliser `image` crate pour I/O.

**Performance** :

- Rendering 60 FPS minimum
- Opérations < 16ms (1 frame)
- Multithreading pour filtres lourds
- Dirty rectangles pour redraw partiel

**Features bonus** :
- Tablet support (pression stylus)
- Brush dynamics
- Color picker avec eyedropper
- Gradient tool
- Clone stamp tool
- Animations (frames)

**Livrables** :
- Application graphique complète
- Galerie d'images créées
- Tutoriel vidéo
- Benchmarks de filtres

---

## Projet 7 : my_shell - Terminal shell interactif (Semaines 11-12)
**Durée** : 14 jours | **Difficulté** : ⭐⭐⭐⭐☆

### Objectifs pédagogiques
- Processus et signaux Unix
- Parsing de commandes complexes
- Job control
- Interaction système bas niveau

### Description
Créer un shell Unix complet avec pipelines, redirections, variables, scripting, et job control comme bash/zsh.

### Fonctionnalités du shell

**Prompt personnalisable** :
```bash
[user@hostname:~/project] (git:main) $
```

**Parsing de commandes** :

Syntaxe à supporter :
```bash
# Commandes simples
ls -la /home

# Pipes
cat file.txt | grep "error" | wc -l

# Redirections
echo "hello" > output.txt
cat < input.txt
command 2> errors.txt
command &> all.txt
command >> append.txt

# Background jobs
long_command &

# Commandes conditionnelles
make && ./program
test -f file || echo "not found"

# Subshells
result=$(cat file.txt)
echo "Files: $(ls | wc -l)"

# Quotes et escaping
echo "Hello $USER"
echo 'Literal $USER'
echo "Path: \"$PATH\""
```

**Architecture** :

```rust
pub struct Shell {
    environment: HashMap<String, String>,
    aliases: HashMap<String, String>,
    jobs: Vec<Job>,
    history: Vec<String>,
    last_exit_code: i32,
}

pub struct Parser {
    lexer: Lexer,
}

pub enum Token {
    Word(String),
    Pipe,
    Redirect(RedirectType),
    Semicolon,
    Ampersand,
    AndAnd,
    OrOr,
    LeftParen,
    RightParen,
    Dollar,
    // ...
}

pub struct Command {
    program: String,
    args: Vec<String>,
    redirects: Vec<Redirect>,
}

pub struct Pipeline {
    commands: Vec<Command>,
    background: bool,
}

pub enum Redirect {
    Input(String),           // < file
    Output(String),          // > file
    Append(String),          // >> file
    Error(String),           // 2> file
    ErrorOutput(String),     // &> file
}

pub struct Job {
    id: usize,
    pipeline: Pipeline,
    pgid: libc::pid_t,
    status: JobStatus,
}

pub enum JobStatus {
    Running,
    Stopped,
    Done,
}
```

**Builtins à implémenter** :

```rust
// Builtins essentiels
cd [PATH]                 // Changer de répertoire
pwd                       // Afficher working directory
echo [ARGS...]            // Afficher texte
export VAR=value          // Exporter variable
unset VAR                 // Supprimer variable
exit [CODE]               // Quitter shell

// Job control
jobs                      // Lister jobs
fg [JOB_ID]              // Ramener en foreground
bg [JOB_ID]              // Continuer en background
kill [-SIGNAL] PID       // Envoyer signal

// Historique
history                   // Afficher historique
!!                        // Répéter dernière commande
!N                        // Répéter commande N
Ctrl+R                    // Recherche historique

// Autres
alias name='command'      // Créer alias
unalias name             // Supprimer alias
type command             // Type de commande
source file              // Exécuter script
```

**Exécution de processus** :

```rust
use nix::unistd::{fork, execvp, ForkResult};
use nix::sys::wait::waitpid;
use nix::sys::signal::{kill, Signal};

impl Shell {
    fn execute_command(&mut self, cmd: &Command) -> Result<i32> {
        match unsafe { fork() } {
            Ok(ForkResult::Parent { child }) => {
                // Parent process
                self.handle_parent(child)
            }
            Ok(ForkResult::Child) => {
                // Child process
                self.setup_child(cmd)?;
                self.exec_child(cmd)?;
                unreachable!()
            }
            Err(e) => Err(e.into()),
        }
    }
    
    fn execute_pipeline(&mut self, pipeline: &Pipeline) -> Result<i32> {
        // Créer pipes entre commandes
        // fork pour chaque commande
        // Setup stdin/stdout avec pipes
        // Wait sur tous les processus
    }
}
```

**Gestion des signaux** :

```rust
// Gérer Ctrl+C, Ctrl+Z, Ctrl+D
use signal_hook::{consts::SIGINT, iterator::Signals};

fn setup_signal_handlers(shell: Arc<Mutex<Shell>>) {
    let mut signals = Signals::new(&[
        SIGINT,   // Ctrl+C
        SIGTSTP,  // Ctrl+Z
        SIGCHLD,  // Enfant terminé
    ]).unwrap();
    
    thread::spawn(move || {
        for sig in signals.forever() {
            match sig {
                SIGINT => {
                    // Interrompre foreground job
                }
                SIGTSTP => {
                    // Suspendre foreground job
                }
                SIGCHLD => {
                    // Reap zombie processes
                }
                _ => {}
            }
        }
    });
}
```

**Variables et expansion** :

```rust
fn expand_variables(&self, input: &str) -> String {
    let mut result = input.to_string();
    
    // $VAR ou ${VAR}
    // $? (exit code)
    // $$ (shell PID)
    // $! (last background PID)
    // $0-$9 (arguments)
    
    // Tilde expansion: ~/path -> /home/user/path
    // Glob expansion: *.rs -> file1.rs file2.rs
    
    result
}
```

**Autocomplétion** :

```rust
use rustyline::{Editor, Config};
use rustyline::completion::{Completer, FilenameCompleter};

struct ShellCompleter {
    file_completer: FilenameCompleter,
    command_completer: CommandCompleter,
}

impl Completer for ShellCompleter {
    fn complete(&self, line: &str, pos: usize) -> Result<(usize, Vec<String>)> {
        // Complétion commandes si début de ligne
        // Complétion fichiers sinon
        // Complétion variables pour $VAR
    }
}
```

**Scripting** :

Support pour fichiers scripts :
```bash
#!/path/to/my_shell

# Commentaires
VAR="value"
echo "Variable: $VAR"

# Conditions
if [ -f "$VAR" ]; then
    echo "File exists"
fi

# Boucles
for file in *.txt; do
    echo "Processing $file"
done

# Fonctions
my_function() {
    echo "Args: $@"
    return 0
}
```

**Configuration** :

Fichiers de config :
- `~/.myshellrc` : chargé au démarrage
- Support pour profils (login shell)

**Features avancées** :
- Vi/Emacs mode pour édition ligne
- Syntax highlighting en temps réel
- Suggestions basées sur historique
- Multi-ligne pour commandes longues
- Bracket matching

**Tests** :
- Tests unitaires pour parser
- Tests d'intégration pour builtins
- Tests de job control
- Tests de gestion signaux

**Livrables** :
- Shell fonctionnel
- Documentation complète
- Scripts d'exemple
- Rapport de compatibilité POSIX
- Benchmark vs bash

---

# 🎯 MOIS 4 : Intelligence artificielle et données

## Projet 8 : my_neural_network - Réseau de neurones from scratch (Semaines 13-15)
**Durée** : 21 jours | **Difficulté** : ⭐⭐⭐⭐⭐

### Objectifs pédagogiques
- Algorithmes de machine learning
- Calcul matriciel et algèbre linéaire
- Optimisation numérique
- Visualisation de données

### Description
Implémenter un framework de réseaux de neurones complet avec backpropagation, différentes architectures, et application à des problèmes réels (MNIST, classification).

### Architecture du framework

**API de haut niveau** :

```rust
use my_nn::prelude::*;

// Créer un réseau
let mut model = Sequential::new()
    .add(Dense::new(784, 128))
    .add(Activation::ReLU)
    .add(Dense::new(128, 64))
    .add(Activation::ReLU)
    .add(Dense::new(64, 10))
    .add(Activation::Softmax);

// Compiler le modèle
model.compile(
    optimizer: Adam::new(learning_rate: 0.001),
    loss: CategoricalCrossentropy,
    metrics: vec![Accuracy],
);

// Entraîner
model.fit(
    x_train,
    y_train,
    epochs: 10,
    batch_size: 32,
    validation_split: 0.2,
);

// Prédire
let predictions = model.predict(x_test);

// Sauvegarder
model.save("model.mynn")?;
```

**Structure interne** :

```rust
// Tenseur N-dimensionnel
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>) -> Self;
    pub fn zeros(shape: Vec<usize>) -> Self;
    pub fn ones(shape: Vec<usize>) -> Self;
    pub fn random_normal(shape: Vec<usize>, mean: f32, std: f32) -> Self;
    
    // Opérations
    pub fn matmul(&self, other: &Tensor) -> Tensor;
    pub fn add(&self, other: &Tensor) -> Tensor;
    pub fn mul(&self, other: &Tensor) -> Tensor;
    pub fn transpose(&self) -> Tensor;
    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor;
    
    // Indexing
    pub fn slice(&self, ranges: &[Range<usize>]) -> Tensor;
    pub fn get(&self, indices: &[usize]) -> f32;
    pub fn set(&mut self, indices: &[usize], value: f32);
}

// Layer trait
pub trait Layer: Send + Sync {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn backward(&mut self, grad_output: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<&Tensor>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
    fn gradients(&self) -> Vec<&Tensor>;
}

// Couche Dense (Fully Connected)
pub struct Dense {
    weights: Tensor,
    bias: Tensor,
    grad_weights: Tensor,
    grad_bias: Tensor,
    last_input: Option<Tensor>,
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        // Initialisation Xavier/He
        let limit = (6.0 / (input_size + output_size) as f32).sqrt();
        let weights = Tensor::random_uniform(
            vec![input_size, output_size],
            -limit,
            limit,
        );
        let bias = Tensor::zeros(vec![output_size]);
        
        Self {
            weights,
            bias,
            grad_weights: Tensor::zeros(vec![input_size, output_size]),
            grad_bias: Tensor::zeros(vec![output_size]),
            last_input: None,
        }
    }
}

impl Layer for Dense {
    fn forward(&self, input: &Tensor) -> Tensor {
        // output = input @ weights + bias
        let output = input.matmul(&self.weights).add(&self.bias);
        self.last_input = Some(input.clone());
        output
    }
    
    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        // Calculer gradients
        let input = self.last_input.as_ref().unwrap();
        
        // grad_weights = input^T @ grad_output
        self.grad_weights = input.transpose().matmul(grad_output);
        
        // grad_bias = sum(grad_output, axis=0)
        self.grad_bias = grad_output.sum(axis: 0);
        
        // grad_input = grad_output @ weights^T
        grad_output.matmul(&self.weights.transpose())
    }
}
```

**Fonctions d'activation** :

```rust
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    LeakyReLU { alpha: f32 },
    ELU { alpha: f32 },
    Swish,
}

impl Layer for Activation {
    fn forward(&self, input: &Tensor) -> Tensor {
        match self {
            Activation::ReLU => input.map(|x| x.max(0.0)),
            Activation::Sigmoid => input.map(|x| 1.0 / (1.0 + (-x).exp())),
            Activation::Tanh => input.map(|x| x.tanh()),
            Activation::Softmax => {
                let exp = input.map(|x| x.exp());
                let sum = exp.sum_axis(1).expand_dims(1);
                exp.div(&sum)
            },
            // ...
        }
    }
    
    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        // Dérivées de chaque activation
    }
}
```

**Fonctions de perte** :

```rust
pub trait Loss {
    fn compute(&self, predicted: &Tensor, target: &Tensor) -> f32;
    fn gradient(&self, predicted: &Tensor, target: &Tensor) -> Tensor;
}

pub struct MeanSquaredError;
impl Loss for MeanSquaredError {
    fn compute(&self, predicted: &Tensor, target: &Tensor) -> f32 {
        predicted.sub(target).pow(2.0).mean()
    }
    
    fn gradient(&self, predicted: &Tensor, target: &Tensor) -> Tensor {
        predicted.sub(target).mul_scalar(2.0 / predicted.len() as f32)
    }
}

pub struct CategoricalCrossentropy;
impl Loss for CategoricalCrossentropy {
    fn compute(&self, predicted: &Tensor, target: &Tensor) -> f32 {
        // -sum(target * log(predicted))
        let epsilon = 1e-7; // Pour stabilité numérique
        let clipped = predicted.clip(epsilon, 1.0 - epsilon);
        -target.mul(&clipped.log()).sum() / predicted.shape()[0] as f32
    }
}
```

**Optimiseurs** :

```rust
pub trait Optimizer {
    fn step(&mut self, parameters: &mut [Tensor], gradients: &[Tensor]);
}

// Stochastic Gradient Descent
pub struct SGD {
    learning_rate: f32,
    momentum: f32,
    velocities: Vec<Tensor>,
}

impl Optimizer for SGD {
    fn step(&mut self, parameters: &mut [Tensor], gradients: &[Tensor]) {
        for (i, (param, grad)) in parameters.iter_mut().zip(gradients).enumerate() {
            if self.momentum > 0.0 {
                let velocity = &mut self.velocities[i];
                *velocity = velocity.mul_scalar(self.momentum).sub(&grad.mul_scalar(self.learning_rate));
                *param = param.add(velocity);
            } else {
                *param = param.sub(&grad.mul_scalar(self.learning_rate));
            }
        }
    }
}

// Adam optimizer
pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: usize,
    m: Vec<Tensor>, // First moment
    v: Vec<Tensor>, // Second moment
}

impl Optimizer for Adam {
    fn step(&mut self, parameters: &mut [Tensor], gradients: &[Tensor]) {
        self.t += 1;
        
        for (i, (param, grad)) in parameters.iter_mut().zip(gradients).enumerate() {
            // m_t = beta1 * m_{t-1} + (1 - beta1) * grad
            self.m[i] = self.m[i].mul_scalar(self.beta1)
                .add(&grad.mul_scalar(1.0 - self.beta1));
            
            // v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
            self.v[i] = self.v[i].mul_scalar(self.beta2)
                .add(&grad.pow(2.0).mul_scalar(1.0 - self.beta2));
            
            // Bias correction
            let m_hat = self.m[i].div_scalar(1.0 - self.beta1.powi(self.t as i32));
            let v_hat = self.v[i].div_scalar(1.0 - self.beta2.powi(self.t as i32));
            
            // Update
            *param = param.sub(&m_hat.div(&v_hat.sqrt().add_scalar(self.epsilon))
                .mul_scalar(self.learning_rate));
        }
    }
}
```

**Couches avancées** :

1. **Dropout** (régularisation) :
```rust
pub struct Dropout {
    rate: f32,
    training: bool,
    mask: Option<Tensor>,
}

impl Layer for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        if self.training {
            let mask = Tensor::random_bernoulli(input.shape(), 1.0 - self.rate);
            self.mask = Some(mask.clone());
            input.mul(&mask).div_scalar(1.0 - self.rate)
        } else {
            input.clone()
        }
    }
}
```

2. **BatchNormalization** :
```rust
pub struct BatchNorm {
    gamma: Tensor,  // Scale
    beta: Tensor,   // Shift
    running_mean: Tensor,
    running_var: Tensor,
    momentum: f32,
    epsilon: f32,
}
```

3. **Convolutional** (CNN) :
```rust
pub struct Conv2D {
    filters: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: Padding,
    weights: Tensor,  // [kernel_h, kernel_w, in_channels, out_channels]
    bias: Tensor,
}

impl Layer for Conv2D {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Implémenter convolution 2D
        // im2col pour efficacité
    }
}
```

4. **MaxPooling** :
```rust
pub struct MaxPool2D {
    pool_size: (usize, usize),
    stride: (usize, usize),
}
```

5. **LSTM/GRU** (pour séquences) :
```rust
pub struct LSTM {
    input_size: usize,
    hidden_size: usize,
    // Gates: forget, input, output
    weights_ih: Tensor,
    weights_hh: Tensor,
    bias: Tensor,
}
```

**Applications pratiques** :

### Application 1 : Classification MNIST

```rust
// Charger MNIST
let (x_train, y_train) = load_mnist_train()?;
let (x_test, y_test) = load_mnist_test()?;

// Normaliser
let x_train = x_train.div_scalar(255.0);
let x_test = x_test.div_scalar(255.0);

// Modèle CNN
let mut model = Sequential::new()
    .add(Conv2D::new(32, (3, 3), activation: ReLU))
    .add(MaxPool2D::new((2, 2)))
    .add(Conv2D::new(64, (3, 3), activation: ReLU))
    .add(MaxPool2D::new((2, 2)))
    .add(Flatten)
    .add(Dense::new(128, activation: ReLU))
    .add(Dropout::new(0.5))
    .add(Dense::new(10, activation: Softmax));

model.compile(
    optimizer: Adam::new(0.001),
    loss: CategoricalCrossentropy,
);

// Training avec callbacks
model.fit(
    x_train, y_train,
    epochs: 20,
    batch_size: 128,
    validation_data: (x_test, y_test),
    callbacks: vec![
        EarlyStopping::new(patience: 3),
        ModelCheckpoint::new("best_model.mynn"),
        TensorBoard::new("./logs"),
    ],
);

// Évaluation
let accuracy = model.evaluate(x_test, y_test);
println!("Test accuracy: {:.2}%", accuracy * 100.0);
```

### Application 2 : Régression (prédiction prix)

```rust
// Dataset Boston Housing ou créer synthetic data
let (x_train, y_train) = load_regression_data()?;

let mut model = Sequential::new()
    .add(Dense::new(13, 64, activation: ReLU))
    .add(Dense::new(64, 32, activation: ReLU))
    .add(Dense::new(32, 1)); // Pas d'activation pour régression

model.compile(
    optimizer: Adam::new(0.01),
    loss: MeanSquaredError,
    metrics: vec![MeanAbsoluteError],
);

model.fit(x_train, y_train, epochs: 100, batch_size: 32);
```

### Application 3 : Générateur de texte (RNN)

```rust
// Modèle LSTM pour génération de texte
let mut model = Sequential::new()
    .add(Embedding::new(vocab_size: 10000, embedding_dim: 256))
    .add(LSTM::new(256, return_sequences: true))
    .add(LSTM::new(256))
    .add(Dense::new(vocab_size, activation: Softmax));

// Entraîner sur Shakespeare, etc.
model.fit(text_sequences, next_chars, epochs: 50);

// Générer du texte
let generated = model.generate(seed_text: "To be or not to", length: 100);
```

**Optimisations** :

1. **SIMD** pour opérations vectorielles :
```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn matmul_simd(a: &[f32], b: &[f32], c: &mut [f32]) {
    // Utiliser AVX2 pour 8x speedup
}
```

2. **Parallélisme** :
```rust
use rayon::prelude::*;

// Paralléliser sur mini-batches
batches.par_iter().for_each(|batch| {
    model.train_on_batch(batch);
});
```

3. **GPU** (bonus) :
```rust
// Utiliser wgpu pour compute shaders
// Ou CUDA bindings si disponible
```

**Visualisation et debugging** :

```rust
// Visualiser architecture
model.summary();
/*
Layer (type)                Output Shape              Param #
=================================================================
dense_1 (Dense)            (None, 128)               100480
activation_1 (ReLU)        (None, 128)               0
dense_2 (Dense)            (None, 10)                1290
=================================================================
Total params: 101,770
Trainable params: 101,770
*/

// Plot training curves
let plotter = TrainingPlotter::new();
plotter.plot_history(&model.history);
// Génère graphique loss/accuracy par epoch

// Visualiser poids
let weights = model.layers[0].weights();
visualize_weights(weights, "weights_layer1.png");

// Visualiser activations
let activations = model.get_activations(sample_input);
visualize_activations(activations);
```

**Features bonus** :
- Transfer learning (charger pré-trained models)
- Data augmentation pour images
- Gradient checking pour debug
- Mixed precision training (FP16)
- Distributed training (multi-GPU)
- AutoML pour hyperparameter tuning
- Model interpretability (LIME, SHAP)

**Livrables** :
- Framework complet my_nn
- 3+ applications démonstration
- Benchmarks vs PyTorch/TensorFlow
- Documentation exhaustive
- Tutoriels et exemples
- Article de blog technique
- Visualisations des résultats

---

# 🎯 MOIS 5 : Réseau et systèmes distribués

## Projet 9 : my_torrent - Client BitTorrent (Semaines 16-17)
**Durée** : 14 jours | **Difficulté** : ⭐⭐⭐⭐☆

### Objectifs pédagogiques
- Protocoles réseau P2P
- Programmation réseau async
- Gestion de fichiers distribués
- Cryptographie appliquée

### Description
Implémenter un client BitTorrent complet suivant le protocole BEP (BitTorrent Enhancement Proposal), capable de télécharger et partager des fichiers.

### Fonctionnalités du protocole

**Architecture** :

```
my_torrent/
├── src/
│   ├── torrent/
│   │   ├── metainfo.rs      // Parser .torrent files
│   │   ├── piece.rs         // Gestion des pièces
│   │   └── file.rs          // Gestion fichiers
│   ├── tracker/
│   │   ├── http.rs          // HTTP tracker
│   │   ├── udp.rs           // UDP tracker
│   │   └── dht.rs           // DHT (Kademlia)
│   ├── peer/
│   │   ├── connection.rs    // Connexion peer
│   │   ├── protocol.rs      // Protocole BitTorrent
│   │   └── manager.rs       // Pool de peers
│   ├── disk/
│   │   ├── io.rs           // Async disk I/O
│   │   └── cache.rs        // Cache de pièces
│   └── ui/
│       ├── tui.rs          // Interface terminal
│       └── web.rs          // Web UI (bonus)
```

**Format .torrent (bencode)** :

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct Torrent {
    pub announce: String,        // Tracker URL
    pub announce_list: Option<Vec<Vec<String>>>,
    pub info: Info,
    pub creation_date: Option<i64>,
    pub comment: Option<String>,
    pub created_by: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Info {
    pub name: String,
    pub piece_length: usize,
    pub pieces: Vec<u8>,         // SHA1 hashes concaténés
    #[serde(flatten)]
    pub mode: Mode,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Mode {
    SingleFile {
        length: usize,
        md5sum: Option<String>,
    },
    MultiFile {
        files: Vec<FileInfo>,
    },
}

#[derive(Debug, Deserialize, Serialize)]
pub struct FileInfo {
    pub length: usize,
    pub path: Vec<String>,
    pub md5sum: Option<String>,
}

// Parser bencode
pub fn parse_torrent(data: &[u8]) -> Result<Torrent> {
    // Implémenter parser bencode
    // Format: d8:announce..e
}
```

**Protocole Tracker** :

```rust
// Requête HTTP tracker
pub struct TrackerRequest {
    pub info_hash: [u8; 20],     // SHA1 du info dict
    pub peer_id: [u8; 20],       // ID unique du client
    pub port: u16,
    pub uploaded: u64,
    pub downloaded: u64,
    pub left: u64,
    pub compact: bool,
    pub event: Option<Event>,
}

pub enum Event {
    Started,
    Completed,
    Stopped,
}

// Réponse tracker
pub struct TrackerResponse {
    pub interval: u32,           // Secondes avant re-announce
    pub peers: Vec<PeerInfo>,
    pub complete: Option<u32>,   // Seeders
    pub incomplete: Option<u32>, // Leechers
}

pub struct PeerInfo {
    pub ip: IpAddr,
    pub port: u16,
    pub peer_id: Option<[u8; 20]>,
}

// Client tracker async
pub struct TrackerClient {
    announce_url: String,
    info_hash: [u8; 20],
    peer_id: [u8; 20],
}

impl TrackerClient {
    pub async fn announce(&self, request: TrackerRequest) 
        -> Result<TrackerResponse> 
    {
        // HTTP GET request
        // Parser réponse bencode
    }
}
```

**Protocole Peer** :

Messages du protocole :
```rust
pub enum Message {
    KeepAlive,
    Choke,
    Unchoke,
    Interested,
    NotInterested,
    Have { piece_index: u32 },
    Bitfield { bitfield: Vec<u8> },
    Request { 
        index: u32, 
        begin: u32, 
        length: u32 
    },
    Piece { 
        index: u32, 
        begin: u32, 
        block: Vec<u8> 
    },
    Cancel { 
        index: u32, 
        begin: u32, 
        length: u32 
    },
    Port { port: u16 },
}

// Handshake
pub struct Handshake {
    pub protocol: [u8; 19],      // "BitTorrent protocol"
    pub reserved: [u8; 8],
    pub info_hash: [u8; 20],
    pub peer_id: [u8; 20],
}

// Connexion peer
pub struct PeerConnection {
    stream: TcpStream,
    peer_id: [u8; 20],
    am_choking: bool,
    am_interested: bool,
    peer_choking: bool,
    peer_interested: bool,
    peer_bitfield: Bitfield,
}

impl PeerConnection {
    pub async fn connect(addr: SocketAddr, info_hash: [u8; 20]) 
        -> Result<Self> 
    {
        let mut stream = TcpStream::connect(addr).await?;
        
        // Envoyer handshake
        let handshake = Handshake::new(info_hash, generate_peer_id());
        stream.write_all(&handshake.to_bytes()).await?;
        
        // Recevoir handshake
        let response = Handshake::from_stream(&mut stream).await?;
        
        // Vérifier info_hash
        if response.info_hash != info_hash {
            return Err("Info hash mismatch");
        }
        
        Ok(Self {
            stream,
            peer_id: response.peer_id,
            // ...
        })
    }
    
    pub async fn send_message(&mut self, msg: Message) -> Result<()> {
        let bytes = msg.to_bytes();
        self.stream.write_all(&bytes).await?;
        Ok(())
    }
    
    pub async fn receive_message(&mut self) -> Result<Message> {
        // Lire longueur (4 bytes)
        let mut len_buf = [0u8; 4];
        self.stream.read_exact(&mut len_buf).await?;
        let len = u32::from_be_bytes(len_buf);
        
        if len == 0 {
            return Ok(Message::KeepAlive);
        }
        
        // Lire message
        let mut buf = vec![0u8; len as usize];
        self.stream.read_exact(&mut buf).await?;
        
        Message::from_bytes(&buf)
    }
}
```

**Gestion des pièces** :

```rust
pub struct PieceManager {
    pieces: Vec<Piece>,
    piece_length: usize,
    total_length: usize,
}

pub struct Piece {
    index: u32,
    hash: [u8; 20],
    blocks: Vec<Block>,
    complete: bool,
}

pub struct Block {
    begin: u32,
    length: u32,
    data: Option<Vec<u8>>,
    requested: bool,
}

impl PieceManager {
    pub fn next_block_to_request(&self, peer_bitfield: &Bitfield) 
        -> Option<(u32, u32, u32)> 
    {
        // Stratégie: rarest first
        // Trouver pièce la plus rare que peer possède
        // Retourner prochain block non-requis
    }
    
    pub fn add_block(&mut self, index: u32, begin: u32, data: Vec<u8>) 
        -> Result<()> 
    {
        let piece = &mut self.pieces[index as usize];
        
        // Ajouter block
        let block_index = begin / BLOCK_SIZE;
        piece.blocks[block_index].data = Some(data);
        
        // Vérifier si pièce complète
        if piece.is_complete() {
            // Vérifier hash SHA1
            let computed_hash = sha1(&piece.data());
            if computed_hash != piece.hash {
                // Hash invalide, re-télécharger
                piece.reset();
                return Err("Hash mismatch");
            }
            piece.complete = true;
        }
        
        Ok(())
    }
}
```

**Stratégies de téléchargement** :

1. **Rarest First** :
```rust
fn select_piece_rarest_first(
    pieces: &[Piece],
    peer_bitfields: &[Bitfield],
) -> Option<usize> {
    // Compter disponibilité de chaque pièce
    let mut availability = vec![0; pieces.len()];
    for bitfield in peer_bitfields {
        for i in 0..pieces.len() {
            if bitfield.has_piece(i) {
                availability[i] += 1;
            }
        }
    }
    
    // Sélectionner pièce la plus rare non-complète
    pieces.iter()
        .enumerate()
        .filter(|(_, p)| !p.complete)
        .min_by_key(|(i, _)| availability[*i])
        .map(|(i, _)| i)
}
```

2. **Endgame Mode** :
```rust
// Quand presque fini, requêter même blocks à plusieurs peers
// Annuler requêtes dès qu'un block arrive
```

**DHT (Distributed Hash Table)** :

```rust
// Implémentation Kademlia DHT pour trouver peers sans tracker

pub struct DhtNode {
    id: [u8; 20],
    routing_table: RoutingTable,
    token_manager: TokenManager,
}

pub struct RoutingTable {
    buckets: Vec<KBucket>,
    our_id: [u8; 20],
}

pub struct KBucket {
    nodes: Vec<NodeInfo>,
    last_changed: Instant,
}

// Messages DHT (Kademlia)
pub enum DhtMessage {
    Ping { id: [u8; 20] },
    FindNode { id: [u8; 20], target: [u8; 20] },
    GetPeers { id: [u8; 20], info_hash: [u8; 20] },
    AnnouncePeer { 
        id: [u8; 20], 
        info_hash: [u8; 20],
        port: u16,
        token: Vec<u8>,
    },
}
```