// Copyright (c) Metacraft Labs
// Parts of the file are Copyright (c) Aptos Foundation
// All Aptos Foundation code and content is licensed pursuant to the Innovation-Enabling Source Code License, available at https://github.com/aptos-labs/aptos-core/blob/main/LICENSE

//! Value-level execution tracing in the Move trace format (version 3).
//!
//! When enabled, the interpreter writes one JSON object per line (NDJSON) describing every
//! frame it opens and closes, every instruction it executes (with its pc and the gas left),
//! and the effects of each instruction on the operand stack and on locals, all with typed
//! values: `OpenFrame` / `Instruction` / `Effect` (`Push`, `Pop`, `Read`, `Write`,
//! `ExecutionError`) / `CloseFrame`. The format is the one the Sui Move VM produces
//! (`move-trace-format`), so tools that consume Sui traces can consume these as well.
//!
//! Tracing is off by default. It is turned on with [`enable_value_tracing`] or by setting the
//! `MOVE_VM_VALUE_TRACE` environment variable to an output path. While it is off, the
//! interpreter checks a single boolean per entrypoint and per executed code unit, and the per
//! instruction hook shares the branch the interpreter already takes for its debugger.
//!
//! The tracer observes the interpreter between instructions: before executing an instruction
//! it records the values the instruction is about to pop; before executing the next one (or
//! when a frame opens or closes) it records what the instruction pushed and which locals it
//! read or wrote. It never holds on to VM values between hooks, so reference counts, and with
//! them the VM's behaviour, are unaffected. While tracing, the interpreter runs with full
//! runtime type checks, which keeps the type of every operand stack slot and local available
//! for rendering values with their struct names and field names.

use crate::{frame::Frame, interpreter::Stack};
use move_binary_format::errors::{PartialVMResult, VMError};
use move_core_types::{
    account_address::AccountAddress,
    int256::{I256, U256},
    language_storage::{StructTag, TypeTag},
    value::{MoveStructLayout, MoveTypeLayout},
    vm_status::StatusCode,
};
use move_vm_types::{
    delayed_values::delayed_field_id::DelayedFieldID,
    instr::Instruction,
    loaded_data::runtime_types::Type,
    values::{AbstractFunction, Value},
    views::{ValueView, ValueVisitor},
};
use std::{
    collections::HashMap,
    fmt::Write as _,
    fs::File,
    io::{self, BufWriter, Write},
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex, Once,
    },
};
use triomphe::Arc as TriompheArc;

/// Environment variable naming the file to write a value trace to.
pub const MOVE_VM_VALUE_TRACE_ENV_VAR_NAME: &str = "MOVE_VM_VALUE_TRACE";

/// The trace format version written in the header line.
const TRACE_FORMAT_VERSION: u32 = 3;

static ENABLED: AtomicBool = AtomicBool::new(false);
static ENV_INIT: Once = Once::new();
static TRACER: Mutex<Option<ValueTracer>> = Mutex::new(None);

/// Start writing a value trace of all subsequent Move execution to `path`, replacing any trace
/// in progress.
pub fn enable_value_tracing(path: &Path) -> io::Result<()> {
    let tracer = ValueTracer::create(path)?;
    let previous = TRACER.lock().unwrap().replace(tracer);
    if let Some(mut previous) = previous {
        previous.out.flush()?;
    }
    ENABLED.store(true, Ordering::Release);
    Ok(())
}

/// Summary of a finished value trace.
#[derive(Debug, Clone, Copy)]
pub struct ValueTraceSummary {
    /// Number of events written after the header line.
    pub events: u64,
    /// Number of `Instruction` events among them.
    pub instructions: u64,
}

/// Stop tracing, flush the trace file and close it. Returns `None` if no trace was in progress.
pub fn finish_value_tracing() -> io::Result<Option<ValueTraceSummary>> {
    ENABLED.store(false, Ordering::Release);
    match TRACER.lock().unwrap().take() {
        Some(mut tracer) => {
            tracer.out.flush()?;
            Ok(Some(ValueTraceSummary {
                events: tracer.next_event,
                instructions: tracer.instructions,
            }))
        },
        None => Ok(None),
    }
}

/// Whether value tracing is on. Reads `MOVE_VM_VALUE_TRACE` the first time it is called.
#[inline]
pub fn is_value_tracing_enabled() -> bool {
    ENV_INIT.call_once(|| {
        if let Ok(path) = std::env::var(MOVE_VM_VALUE_TRACE_ENV_VAR_NAME) {
            if !path.is_empty() {
                if let Err(err) = enable_value_tracing(Path::new(&path)) {
                    eprintln!("cannot write Move value trace to {}: {}", path, err);
                }
            }
        }
    });
    ENABLED.load(Ordering::Relaxed)
}

/// Gives the tracer the types it needs to render values, without touching gas or the
/// transaction's traversal context.
pub(crate) trait TraceTypeResolver {
    /// The fully annotated layout (struct tags, field and variant names) of `ty`.
    fn annotated_layout(&self, ty: &Type) -> Option<TriompheArc<MoveTypeLayout>>;
    /// The type tag of `ty`.
    fn type_tag(&self, ty: &Type) -> Option<TypeTag>;
    /// `ty` with its type parameters replaced by `ty_args`.
    fn instantiate(&self, ty: &Type, ty_args: &[Type]) -> Option<Type>;
}

/// The interpreter state visible to a tracing hook.
pub(crate) struct TraceCtx<'a> {
    pub(crate) resolver: &'a dyn TraceTypeResolver,
    pub(crate) stack: &'a Stack,
    /// Frames of the callers of `current`, outermost first.
    pub(crate) callers: &'a [Frame],
    pub(crate) current: &'a Frame,
    pub(crate) gas_left: u64,
}

/// Run `f` on the active tracer, if any.
#[inline]
fn with_tracer(f: impl FnOnce(&mut ValueTracer)) {
    if let Some(tracer) = TRACER.lock().unwrap().as_mut() {
        f(tracer)
    }
}

/// Hook: an entrypoint's (or a Move callee's) frame has just been created and is `ctx.current`.
pub(crate) fn open_frame(ctx: &TraceCtx) {
    with_tracer(|t| t.open_frame(ctx))
}

/// Hook: `ctx.current` executed `Ret` and its return values are on top of the stack.
pub(crate) fn close_frame(ctx: &TraceCtx) {
    with_tracer(|t| t.close_frame(ctx))
}

/// Hook: `instruction` at `pc` of `ctx.current` is about to execute.
pub(crate) fn before_instruction(ctx: &TraceCtx, instruction: &Instruction) {
    with_tracer(|t| t.before_instruction(ctx, instruction))
}

/// Hook: native `function` is about to be called with `num_stack_args` arguments taken from
/// the top of the stack.
pub(crate) fn open_native(ctx: &TraceCtx, function: &crate::LoadedFunction, num_stack_args: usize) {
    with_tracer(|t| t.open_native(ctx, function, num_stack_args))
}

/// Hook: the native opened by [`open_native`] returned. When it dispatched to a Move function,
/// `ctx.current` is that function's new frame.
pub(crate) fn close_native(ctx: &TraceCtx, function: &crate::LoadedFunction, dispatched: bool) {
    with_tracer(|t| t.close_native(ctx, function, dispatched))
}

/// Hook: an entrypoint finished, successfully or not.
pub(crate) fn finish_entrypoint<T>(result: &Result<T, VMError>, gas_left: u64) {
    with_tracer(|t| t.finish_entrypoint(result.as_ref().err(), gas_left))
}

/// Where a reference points: a local of a frame (identified by the trace index of its
/// `OpenFrame`), a field or element of another location, or storage.
#[derive(Clone, Debug)]
enum Loc {
    Local(u64, u64),
    Indexed(Box<Loc>, u64),
    Global(u64),
}

impl Loc {
    fn root(&self) -> &Loc {
        match self {
            Loc::Indexed(inner, _) => inner.root(),
            other => other,
        }
    }

    fn write_json(&self, out: &mut String) {
        match self {
            Loc::Local(frame, idx) => {
                let _ = write!(out, "{{\"Local\":[{},{}]}}", frame, idx);
            },
            Loc::Indexed(inner, idx) => {
                out.push_str("{\"Indexed\":[");
                inner.write_json(out);
                let _ = write!(out, ",{}]}}", idx);
            },
            Loc::Global(idx) => {
                let _ = write!(out, "{{\"Global\":{}}}", idx);
            },
        }
    }
}

/// What the tracer needs to remember about an instruction between its two hooks.
enum Op {
    CopyLoc(usize),
    MoveLoc(usize),
    StLoc(usize),
    BorrowLoc(usize),
    BorrowField(u64),
    VecBorrow(Option<u64>),
    BorrowGlobal,
    FreezeRef,
    ReadRef,
    WriteRef,
    Abort,
    Other,
}

struct Popped {
    loc: Option<Loc>,
    /// The value as a rendered `TraceValue`.
    json: String,
}

struct Pending {
    op: Op,
    /// Values popped, top of stack first.
    pops: Vec<Popped>,
    /// Stack height once the pops are done.
    base: usize,
}

struct TracedFrame {
    frame_id: u64,
    /// Locations referenced by the locals that hold references.
    local_locs: Vec<Option<Loc>>,
}

struct OpenNative {
    frame_id: u64,
    base: usize,
}

struct ValueTracer {
    out: BufWriter<File>,
    next_event: u64,
    instructions: u64,
    frames: Vec<TracedFrame>,
    native: Option<OpenNative>,
    /// The location each operand stack slot refers to, when it is a reference.
    shadow: Vec<Option<Loc>>,
    pending: Option<Pending>,
    layouts: HashMap<Type, Option<TriompheArc<MoveTypeLayout>>>,
}

impl ValueTracer {
    fn create(path: &Path) -> io::Result<Self> {
        let mut out = BufWriter::with_capacity(4 << 20, File::create(path)?);
        writeln!(out, "{{\"version\":{}}}", TRACE_FORMAT_VERSION)?;
        Ok(Self {
            out,
            next_event: 0,
            instructions: 0,
            frames: vec![],
            native: None,
            shadow: vec![],
            pending: None,
            layouts: HashMap::new(),
        })
    }

    /// Write one event line; returns its trace index.
    fn emit(&mut self, line: &str) -> u64 {
        let idx = self.next_event;
        self.next_event += 1;
        if let Err(err) = self
            .out
            .write_all(line.as_bytes())
            .and_then(|_| self.out.write_all(b"\n"))
        {
            eprintln!("cannot write Move value trace: {}", err);
        }
        idx
    }

    fn emit_effect(&mut self, effect: &str) {
        let line = format!("{{\"Effect\":{}}}", effect);
        self.emit(&line);
    }

    fn layout(
        &mut self,
        resolver: &dyn TraceTypeResolver,
        ty: &Type,
    ) -> Option<TriompheArc<MoveTypeLayout>> {
        if let Some(cached) = self.layouts.get(ty) {
            return cached.clone();
        }
        let layout = resolver.annotated_layout(ty);
        self.layouts.insert(ty.clone(), layout.clone());
        layout
    }

    fn sync_shadow(&mut self, len: usize) {
        self.shadow.resize(len, None);
    }

    // ---------------------------------------------------------------------------------------
    // Value rendering
    // ---------------------------------------------------------------------------------------

    /// Render a non-reference value as a `SerializableMoveValue`.
    fn render_plain(
        &mut self,
        resolver: &dyn TraceTypeResolver,
        node: Option<&Node>,
        ty: Option<&Type>,
    ) -> String {
        let layout = ty.and_then(|ty| self.layout(resolver, ty));
        let mut out = String::new();
        match node {
            Some(node) => write_value(&mut out, strip_ref(node), layout.as_deref()),
            None => write_unavailable(&mut out),
        }
        out
    }

    /// The type of local `idx` of `frame`, when runtime type checks keep it.
    fn local_ty(frame: &Frame, idx: usize) -> Option<&Type> {
        if matches!(frame.local_tys, crate::frame::LocalTys::None)
            || idx >= frame.function.local_tys().len()
        {
            None
        } else {
            Some(frame.local_ty_at(idx))
        }
    }

    /// The frame that owns traced frame `frame_id`, if it is still on the call stack.
    fn find_frame<'a>(&self, ctx: &TraceCtx<'a>, frame_id: u64) -> Option<&'a Frame> {
        let pos = self.frames.iter().rposition(|f| f.frame_id == frame_id)?;
        if pos + 1 == self.frames.len() {
            Some(ctx.current)
        } else if ctx.callers.len() + 1 == self.frames.len() {
            ctx.callers.get(pos)
        } else {
            None
        }
    }

    /// Render the root value of `loc` (a whole local), when it is a local still on the stack.
    fn render_root(&mut self, ctx: &TraceCtx, loc: &Loc) -> Option<String> {
        let Loc::Local(frame_id, idx) = loc.root() else {
            return None;
        };
        let frame = self.find_frame(ctx, *frame_id)?;
        let idx = *idx as usize;
        let node = local_tree(frame, idx)?;
        let ty = Self::local_ty(frame, idx).cloned();
        let layout = ty.as_ref().and_then(|ty| self.layout(ctx.resolver, ty));
        let mut out = String::new();
        write_value(&mut out, strip_ref(&node), layout.as_deref());
        Some(out)
    }

    /// Render a value as a `TraceValue`: `RuntimeValue`, or `ImmRef` / `MutRef` with the
    /// location referenced and a snapshot of the root value there.
    fn render_trace_value(
        &mut self,
        ctx: &TraceCtx,
        node: Option<&Node>,
        is_ref: bool,
        ty: Option<&Type>,
        loc: Option<&Loc>,
    ) -> String {
        let (is_mut, inner_ty) = match ty {
            Some(Type::MutableReference(inner)) => (true, Some(inner.as_ref())),
            Some(Type::Reference(inner)) => (false, Some(inner.as_ref())),
            other => (false, other),
        };
        let is_ref = is_ref || matches!(ty, Some(Type::Reference(_) | Type::MutableReference(_)));
        if !is_ref {
            return format!(
                "{{\"RuntimeValue\":{{\"value\":{}}}}}",
                self.render_plain(ctx.resolver, node, inner_ty)
            );
        }
        let snapshot = loc
            .and_then(|loc| self.render_root(ctx, loc))
            .unwrap_or_else(|| self.render_plain(ctx.resolver, node, inner_ty));
        let mut out = String::new();
        let _ = write!(
            out,
            "{{\"{}\":{{\"location\":",
            if is_mut { "MutRef" } else { "ImmRef" }
        );
        match loc {
            Some(loc) => loc.write_json(&mut out),
            // A reference whose origin the tracer did not see (e.g. one returned by a
            // native): name it after the event that introduces it.
            None => Loc::Global(self.next_event).write_json(&mut out),
        }
        let _ = write!(out, ",\"snapshot\":{}}}}}", snapshot);
        out
    }

    fn render_stack_slot(&mut self, ctx: &TraceCtx, slot: usize, loc: Option<&Loc>) -> String {
        let value = &ctx.stack.value[slot];
        let ty = if ctx.stack.types.len() == ctx.stack.value.len() {
            ctx.stack.types.get(slot)
        } else {
            None
        };
        let node = build_tree(value);
        let is_ref = node.as_ref().is_some_and(Node::is_ref);
        self.render_trace_value(ctx, node.as_ref(), is_ref, ty, loc)
    }

    fn render_local(
        &mut self,
        ctx: &TraceCtx,
        frame: &Frame,
        idx: usize,
        loc: Option<&Loc>,
    ) -> Option<String> {
        let node = local_tree(frame, idx)?;
        let ty = Self::local_ty(frame, idx).cloned();
        Some(self.render_trace_value(ctx, Some(&node), node.is_ref(), ty.as_ref(), loc))
    }

    fn type_json(&self, resolver: &dyn TraceTypeResolver, ty: &Type) -> String {
        let mut out = String::new();
        match resolver.type_tag(ty) {
            Some(tag) => write_type_tag(&mut out, &tag),
            None => write_json_str(&mut out, "?"),
        }
        out
    }

    fn local_type_json(&self, resolver: &dyn TraceTypeResolver, ty: &Type) -> String {
        let (inner, ref_type) = match ty {
            Type::Reference(inner) => (inner.as_ref(), "\"Imm\""),
            Type::MutableReference(inner) => (inner.as_ref(), "\"Mut\""),
            other => (other, "null"),
        };
        format!(
            "{{\"type_\":{},\"ref_type\":{}}}",
            self.type_json(resolver, inner),
            ref_type
        )
    }

    // ---------------------------------------------------------------------------------------
    // Frames
    // ---------------------------------------------------------------------------------------

    fn frame_header(
        &self,
        resolver: &dyn TraceTypeResolver,
        frame_id: u64,
        function: &crate::LoadedFunction,
    ) -> String {
        let module = function.module_or_script_id();
        let address = module.address().to_hex();
        let mut out = String::new();
        let _ = write!(out, "{{\"frame_id\":{},\"function_name\":", frame_id);
        write_json_str(&mut out, function.name());
        out.push_str(",\"module\":{\"address\":");
        write_json_str(&mut out, &address);
        out.push_str(",\"name\":");
        write_json_str(&mut out, module.name().as_str());
        out.push_str("},\"version_id\":");
        write_json_str(&mut out, &address);
        let _ = write!(
            out,
            ",\"binary_member_index\":{},\"type_instantiation\":[",
            function.index().0
        );
        for (i, ty) in function.ty_args().iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&self.type_json(resolver, ty));
        }
        out.push_str("],\"return_types\":[");
        for (i, ty) in function.return_tys().iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let ty = if function.ty_args().is_empty() {
                Some(ty.clone())
            } else {
                resolver.instantiate(ty, function.ty_args())
            };
            match ty {
                Some(ty) => out.push_str(&self.local_type_json(resolver, &ty)),
                None => out.push_str("{\"type_\":\"?\",\"ref_type\":null}"),
            }
        }
        out.push(']');
        out
    }

    fn open_frame(&mut self, ctx: &TraceCtx) {
        let frame = ctx.current;
        let function = frame.function.as_ref();
        // The instruction that made the call is consumed by the call; its arguments are now
        // the callee's parameters.
        self.pending = None;
        let num_params = function.param_tys().len();
        let split = ctx.stack.value.len().min(self.shadow.len());
        let mut arg_locs = self.shadow.split_off(split);
        self.sync_shadow(ctx.stack.value.len());
        let skip = arg_locs.len().saturating_sub(num_params);
        arg_locs.drain(..skip);
        while arg_locs.len() < num_params {
            arg_locs.insert(0, None);
        }

        let frame_id = self.next_event;
        let num_locals = function.local_tys().len();
        let mut local_locs = vec![None; num_locals];
        for (i, loc) in arg_locs.into_iter().enumerate().take(num_locals) {
            local_locs[i] = loc;
        }
        self.frames.push(TracedFrame {
            frame_id,
            local_locs,
        });

        let mut out = String::from("{\"OpenFrame\":{\"frame\":");
        out.push_str(&self.frame_header(ctx.resolver, frame_id, function));
        out.push_str(",\"parameters\":[");
        for i in 0..num_params {
            if i > 0 {
                out.push(',');
            }
            let loc = self
                .frames
                .last()
                .and_then(|f| f.local_locs.get(i).cloned().flatten());
            let rendered = self
                .render_local(ctx, frame, i, loc.as_ref())
                .unwrap_or_else(|| {
                    let mut s = String::from("{\"RuntimeValue\":{\"value\":");
                    write_unavailable(&mut s);
                    s.push_str("}}");
                    s
                });
            out.push_str(&rendered);
        }
        out.push_str("],\"locals_types\":[");
        for i in 0..num_locals {
            if i > 0 {
                out.push(',');
            }
            match Self::local_ty(frame, i) {
                Some(ty) => out.push_str(&self.local_type_json(ctx.resolver, ty)),
                None => out.push_str("{\"type_\":\"?\",\"ref_type\":null}"),
            }
        }
        let _ = write!(
            out,
            "],\"is_native\":false}},\"gas_left\":{}}}}}",
            ctx.gas_left
        );
        self.emit(&out);
    }

    /// Render the `n` values on top of the stack, bottom first, with their locations.
    fn render_top(&mut self, ctx: &TraceCtx, n: usize) -> Vec<(Option<Loc>, String)> {
        let len = ctx.stack.value.len();
        let n = n.min(len);
        self.sync_shadow(len);
        (len - n..len)
            .map(|slot| {
                let loc = self.shadow[slot].clone();
                let json = self.render_stack_slot(ctx, slot, loc.as_ref());
                (loc, json)
            })
            .collect()
    }

    fn emit_close(&mut self, frame_id: u64, returns: &[(Option<Loc>, String)], gas_left: u64) {
        let mut out = String::new();
        let _ = write!(
            out,
            "{{\"CloseFrame\":{{\"frame_id\":{},\"return_\":[",
            frame_id
        );
        for (i, (_, json)) in returns.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(json);
        }
        let _ = write!(out, "],\"gas_left\":{}}}}}", gas_left);
        self.emit(&out);
    }

    fn close_frame(&mut self, ctx: &TraceCtx) {
        self.pending = None;
        let num_returns = ctx.current.function.return_tys().len();
        let returns = self.render_top(ctx, num_returns);
        let Some(frame) = self.frames.pop() else {
            return;
        };
        self.emit_close(frame.frame_id, &returns, ctx.gas_left);
        if self.frames.is_empty() {
            self.shadow.clear();
            return;
        }
        // The caller's `Call` instruction pushes the callee's return values.
        for (_, json) in &returns {
            self.emit_effect(&format!("{{\"Push\":{}}}", json));
        }
    }

    fn open_native(
        &mut self,
        ctx: &TraceCtx,
        function: &crate::LoadedFunction,
        num_stack_args: usize,
    ) {
        self.pending = None;
        let args = self.render_top(ctx, num_stack_args);
        let frame_id = self.next_event;
        let mut out = String::from("{\"OpenFrame\":{\"frame\":");
        out.push_str(&self.frame_header(ctx.resolver, frame_id, function));
        out.push_str(",\"parameters\":[");
        for (i, (_, json)) in args.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(json);
        }
        let _ = write!(
            out,
            "],\"locals_types\":[],\"is_native\":true}},\"gas_left\":{}}}}}",
            ctx.gas_left
        );
        self.emit(&out);
        self.native = Some(OpenNative {
            frame_id,
            base: ctx.stack.value.len().saturating_sub(args.len()),
        });
    }

    fn close_native(&mut self, ctx: &TraceCtx, function: &crate::LoadedFunction, dispatched: bool) {
        let Some(native) = self.native.take() else {
            return;
        };
        if dispatched {
            // The native handed control to a Move function, which returns to the caller in
            // its place: close the native here and open the function's frame.
            self.emit_close(native.frame_id, &[], ctx.gas_left);
            self.shadow.truncate(native.base.min(self.shadow.len()));
            self.open_frame(ctx);
            return;
        }
        self.shadow.truncate(native.base.min(self.shadow.len()));
        let returns = self.render_top(ctx, function.return_tys().len());
        self.emit_close(native.frame_id, &returns, ctx.gas_left);
        for (_, json) in &returns {
            self.emit_effect(&format!("{{\"Push\":{}}}", json));
        }
    }

    fn finish_entrypoint(&mut self, error: Option<&VMError>, gas_left: u64) {
        if let Some(error) = error {
            if let Some(pending) = self.pending.take() {
                // An abort pops its code before failing; other instructions fail before their
                // effects take place.
                if matches!(pending.op, Op::Abort) {
                    for popped in &pending.pops {
                        let effect = format!("{{\"Pop\":{}}}", popped.json);
                        self.emit_effect(&effect);
                    }
                }
            }
            let message = match error.major_status() {
                StatusCode::ABORTED => "ABORTED".to_string(),
                status => match error.sub_status() {
                    Some(sub) => format!("{:?} (sub status {})", status, sub),
                    None => format!("{:?}", status),
                },
            };
            let mut effect = String::from("{\"ExecutionError\":");
            write_json_str(&mut effect, &message);
            effect.push('}');
            self.emit_effect(&effect);
        }
        self.pending = None;
        if let Some(native) = self.native.take() {
            self.emit_close(native.frame_id, &[], gas_left);
        }
        while let Some(frame) = self.frames.pop() {
            self.emit_close(frame.frame_id, &[], gas_left);
        }
        self.shadow.clear();
        let _ = self.out.flush();
    }

    // ---------------------------------------------------------------------------------------
    // Instructions
    // ---------------------------------------------------------------------------------------

    fn before_instruction(&mut self, ctx: &TraceCtx, instruction: &Instruction) {
        if self.frames.is_empty() {
            return;
        }
        self.finish_pending(ctx);

        let frame = ctx.current;
        let mut line = String::new();
        let _ = write!(
            line,
            "{{\"Instruction\":{{\"type_parameters\":[],\"pc\":{},\"gas_left\":{},\"instruction\":",
            frame.pc, ctx.gas_left
        );
        write_json_str(&mut line, &instruction.name().to_ascii_uppercase());
        line.push_str("}}");
        self.emit(&line);
        self.instructions += 1;

        let (num_pops, op) = classify(instruction, frame, ctx.stack);
        let len = ctx.stack.value.len();
        let num_pops = num_pops.min(len);
        self.sync_shadow(len);
        let mut pops = Vec::with_capacity(num_pops);
        for slot in (len - num_pops..len).rev() {
            let loc = self.shadow[slot].clone();
            let json = self.render_stack_slot(ctx, slot, loc.as_ref());
            pops.push(Popped { loc, json });
        }
        self.pending = Some(Pending {
            op,
            pops,
            base: len - num_pops,
        });
    }

    fn finish_pending(&mut self, ctx: &TraceCtx) {
        let Some(pending) = self.pending.take() else {
            return;
        };
        let Some(frame_id) = self.frames.last().map(|f| f.frame_id) else {
            return;
        };
        let frame = ctx.current;

        for popped in &pending.pops {
            let effect = format!("{{\"Pop\":{}}}", popped.json);
            self.emit_effect(&effect);
        }

        let len = ctx.stack.value.len();
        let base = pending.base.min(len);
        self.shadow.resize(base, None);
        let first_pop_loc = pending.pops.first().and_then(|p| p.loc.clone());

        // The location the pushed value refers to, when the instruction makes a reference.
        let mut pushed_loc: Option<Loc> = None;
        match &pending.op {
            Op::CopyLoc(idx) | Op::MoveLoc(idx) => {
                let moved = matches!(pending.op, Op::MoveLoc(_));
                let local_loc = self
                    .frames
                    .last()
                    .and_then(|f| f.local_locs.get(*idx).cloned().flatten());
                if len > base {
                    let value = self.render_stack_slot(ctx, base, local_loc.as_ref());
                    let mut effect = String::from("{\"Read\":{\"location\":");
                    Loc::Local(frame_id, *idx as u64).write_json(&mut effect);
                    let _ = write!(
                        effect,
                        ",\"root_value_read\":{},\"moved\":{}}}}}",
                        value, moved
                    );
                    self.emit_effect(&effect);
                }
                pushed_loc = local_loc;
                if moved {
                    if let Some(slot) = self
                        .frames
                        .last_mut()
                        .and_then(|f| f.local_locs.get_mut(*idx))
                    {
                        *slot = None;
                    }
                }
            },
            Op::StLoc(idx) => {
                if let Some(slot) = self
                    .frames
                    .last_mut()
                    .and_then(|f| f.local_locs.get_mut(*idx))
                {
                    *slot = first_pop_loc.clone();
                }
                if let Some(value) = self.render_local(ctx, frame, *idx, first_pop_loc.as_ref()) {
                    let mut effect = String::from("{\"Write\":{\"location\":");
                    Loc::Local(frame_id, *idx as u64).write_json(&mut effect);
                    let _ = write!(effect, ",\"root_value_after_write\":{}}}}}", value);
                    self.emit_effect(&effect);
                }
            },
            Op::BorrowLoc(idx) => {
                let loc = Loc::Local(frame_id, *idx as u64);
                if let Some(value) = self.render_local(ctx, frame, *idx, None) {
                    let mut effect = String::from("{\"Read\":{\"location\":");
                    loc.write_json(&mut effect);
                    let _ = write!(effect, ",\"root_value_read\":{},\"moved\":false}}}}", value);
                    self.emit_effect(&effect);
                }
                pushed_loc = Some(loc);
            },
            Op::BorrowField(offset) => {
                pushed_loc = first_pop_loc.map(|l| Loc::Indexed(Box::new(l), *offset));
            },
            Op::VecBorrow(index) => {
                // Pops are the index (top) and the vector reference.
                let vec_loc = pending.pops.get(1).and_then(|p| p.loc.clone());
                pushed_loc = match (vec_loc, index) {
                    (Some(l), Some(i)) => Some(Loc::Indexed(Box::new(l), *i)),
                    _ => None,
                };
            },
            Op::BorrowGlobal => {
                // Named after the `Push` event emitted below.
                pushed_loc = Some(Loc::Global(self.next_event));
            },
            Op::FreezeRef => pushed_loc = first_pop_loc,
            Op::ReadRef => {
                if let Some(loc) = first_pop_loc {
                    let root = self
                        .render_root(ctx, &loc)
                        .map(|v| format!("{{\"RuntimeValue\":{{\"value\":{}}}}}", v));
                    let root = match root {
                        Some(root) => Some(root),
                        None if len > base => Some(self.render_stack_slot(ctx, base, None)),
                        None => None,
                    };
                    if let Some(root) = root {
                        let mut effect = String::from("{\"Read\":{\"location\":");
                        loc.write_json(&mut effect);
                        let _ = write!(effect, ",\"root_value_read\":{},\"moved\":false}}}}", root);
                        self.emit_effect(&effect);
                    }
                }
            },
            Op::WriteRef => {
                if let Some(loc) = first_pop_loc {
                    // Storage roots are not reachable from the frame: report the value written.
                    let root = self
                        .render_root(ctx, &loc)
                        .map(|v| format!("{{\"RuntimeValue\":{{\"value\":{}}}}}", v))
                        .or_else(|| pending.pops.get(1).map(|p| p.json.clone()));
                    if let Some(root) = root {
                        let mut effect = String::from("{\"Write\":{\"location\":");
                        loc.write_json(&mut effect);
                        let _ = write!(effect, ",\"root_value_after_write\":{}}}}}", root);
                        self.emit_effect(&effect);
                    }
                }
            },
            Op::Abort | Op::Other => {},
        }

        for slot in base..len {
            let loc = if slot == base {
                pushed_loc.clone()
            } else {
                None
            };
            let json = self.render_stack_slot(ctx, slot, loc.as_ref());
            self.emit_effect(&format!("{{\"Push\":{}}}", json));
            self.shadow.push(loc);
        }
    }
}

/// How many values `instruction` pops, and what the tracer needs to remember about it.
/// Calls are not counted: their arguments become the callee's parameters.
fn classify(instruction: &Instruction, frame: &Frame, stack: &Stack) -> (usize, Op) {
    use Instruction as I;
    match instruction {
        I::Ret | I::Branch(_) | I::Nop => (0, Op::Other),
        I::Call(_) | I::CallGeneric(_) | I::CallClosure(_) => (0, Op::Other),
        I::LdU8(_)
        | I::LdU16(_)
        | I::LdU32(_)
        | I::LdU64(_)
        | I::LdU128(_)
        | I::LdU256(_)
        | I::LdI8(_)
        | I::LdI16(_)
        | I::LdI32(_)
        | I::LdI64(_)
        | I::LdI128(_)
        | I::LdI256(_)
        | I::LdConst(_)
        | I::LdTrue
        | I::LdFalse => (0, Op::Other),
        I::CopyLoc(idx) => (0, Op::CopyLoc(*idx as usize)),
        I::MoveLoc(idx) => (0, Op::MoveLoc(*idx as usize)),
        I::StLoc(idx) => (1, Op::StLoc(*idx as usize)),
        I::MutBorrowLoc(idx) | I::ImmBorrowLoc(idx) => (0, Op::BorrowLoc(*idx as usize)),
        I::MutBorrowField(idx) | I::ImmBorrowField(idx) => {
            (1, Op::BorrowField(frame.field_offset(*idx) as u64))
        },
        I::MutBorrowFieldGeneric(idx) | I::ImmBorrowFieldGeneric(idx) => (
            1,
            Op::BorrowField(frame.field_instantiation_offset(*idx) as u64),
        ),
        I::MutBorrowVariantField(idx) | I::ImmBorrowVariantField(idx) => (
            1,
            Op::BorrowField(frame.variant_field_info_at(*idx).offset as u64),
        ),
        I::MutBorrowVariantFieldGeneric(idx) | I::ImmBorrowVariantFieldGeneric(idx) => (
            1,
            Op::BorrowField(frame.variant_field_instantiation_info_at(*idx).offset as u64),
        ),
        I::VecImmBorrow(_) | I::VecMutBorrow(_) => {
            let index = match stack.value.last() {
                Some(Value::U64(i)) => Some(*i),
                _ => None,
            };
            (2, Op::VecBorrow(index))
        },
        I::MutBorrowGlobal(_)
        | I::MutBorrowGlobalGeneric(_)
        | I::ImmBorrowGlobal(_)
        | I::ImmBorrowGlobalGeneric(_) => (1, Op::BorrowGlobal),
        I::FreezeRef => (1, Op::FreezeRef),
        I::ReadRef => (1, Op::ReadRef),
        I::WriteRef => (2, Op::WriteRef),
        I::Abort => (1, Op::Abort),
        I::AbortMsg => (2, Op::Abort),
        I::Pop
        | I::BrTrue(_)
        | I::BrFalse(_)
        | I::Unpack(_)
        | I::UnpackGeneric(_)
        | I::UnpackVariant(_)
        | I::UnpackVariantGeneric(_)
        | I::TestVariant(_)
        | I::TestVariantGeneric(_)
        | I::CastU8
        | I::CastU16
        | I::CastU32
        | I::CastU64
        | I::CastU128
        | I::CastU256
        | I::CastI8
        | I::CastI16
        | I::CastI32
        | I::CastI64
        | I::CastI128
        | I::CastI256
        | I::Not
        | I::Negate
        | I::Exists(_)
        | I::ExistsGeneric(_)
        | I::MoveFrom(_)
        | I::MoveFromGeneric(_)
        | I::VecLen(_)
        | I::VecPopBack(_)
        | I::VecUnpack(_, _) => (1, Op::Other),
        I::Add
        | I::Sub
        | I::Mul
        | I::Mod
        | I::Div
        | I::BitOr
        | I::BitAnd
        | I::Xor
        | I::Or
        | I::And
        | I::Eq
        | I::Neq
        | I::Lt
        | I::Gt
        | I::Le
        | I::Ge
        | I::Shl
        | I::Shr
        | I::MoveTo(_)
        | I::MoveToGeneric(_)
        | I::VecPushBack(_) => (2, Op::Other),
        I::VecSwap(_) => (3, Op::Other),
        I::VecPack(_, n) => (*n as usize, Op::Other),
        I::Pack(idx) => (frame.field_count(*idx) as usize, Op::Other),
        I::PackGeneric(idx) => (frame.field_instantiation_count(*idx) as usize, Op::Other),
        I::PackVariant(idx) => (
            frame.get_struct_variant_at(*idx).field_count as usize,
            Op::Other,
        ),
        I::PackVariantGeneric(idx) => (
            frame.get_struct_variant_instantiation_at(*idx).field_count as usize,
            Op::Other,
        ),
        I::PackClosure(_, mask) | I::PackClosureGeneric(_, mask) => {
            (mask.captured_count() as usize, Op::Other)
        },
    }
}

// -------------------------------------------------------------------------------------------
// Value trees
// -------------------------------------------------------------------------------------------

/// A value as the VM's value visitor reports it, without types.
enum Node {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
    U256(U256),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    I128(i128),
    I256(I256),
    Bool(bool),
    Address(AccountAddress),
    Struct(Vec<Node>),
    Vector(Vec<Node>),
    Closure(String, Vec<Node>),
    Delayed(u64),
    Ref(Box<Node>),
}

impl Node {
    fn is_ref(&self) -> bool {
        matches!(self, Node::Ref(_))
    }
}

enum Open {
    Struct,
    Vector,
    Closure(String),
    Ref,
}

#[derive(Default)]
struct TreeBuilder {
    open: Vec<(Open, usize, Vec<Node>)>,
    done: Option<Node>,
}

impl TreeBuilder {
    fn close(kind: Open, kids: Vec<Node>) -> Node {
        match kind {
            Open::Struct => Node::Struct(kids),
            Open::Vector => Node::Vector(kids),
            Open::Closure(name) => Node::Closure(name, kids),
            Open::Ref => Node::Ref(Box::new(
                kids.into_iter().next().unwrap_or(Node::Struct(vec![])),
            )),
        }
    }

    fn attach(&mut self, mut node: Node) {
        loop {
            match self.open.last_mut() {
                None => {
                    self.done = Some(node);
                    return;
                },
                Some((_, expected, kids)) => {
                    kids.push(node);
                    if kids.len() < *expected {
                        return;
                    }
                    let (kind, _, kids) = self.open.pop().unwrap();
                    node = Self::close(kind, kids);
                },
            }
        }
    }

    fn start(&mut self, kind: Open, len: usize) {
        if len == 0 {
            self.attach(Self::close(kind, vec![]));
        } else {
            self.open
                .push((kind, len, Vec::with_capacity(len.min(1024))));
        }
    }
}

impl ValueVisitor for TreeBuilder {
    fn visit_delayed(&mut self, _depth: u64, id: DelayedFieldID) -> PartialVMResult<()> {
        self.attach(Node::Delayed(id.as_u64()));
        Ok(())
    }

    fn visit_u8(&mut self, _depth: u64, val: u8) -> PartialVMResult<()> {
        self.attach(Node::U8(val));
        Ok(())
    }

    fn visit_u16(&mut self, _depth: u64, val: u16) -> PartialVMResult<()> {
        self.attach(Node::U16(val));
        Ok(())
    }

    fn visit_u32(&mut self, _depth: u64, val: u32) -> PartialVMResult<()> {
        self.attach(Node::U32(val));
        Ok(())
    }

    fn visit_u64(&mut self, _depth: u64, val: u64) -> PartialVMResult<()> {
        self.attach(Node::U64(val));
        Ok(())
    }

    fn visit_u128(&mut self, _depth: u64, val: u128) -> PartialVMResult<()> {
        self.attach(Node::U128(val));
        Ok(())
    }

    fn visit_u256(&mut self, _depth: u64, val: &U256) -> PartialVMResult<()> {
        self.attach(Node::U256(*val));
        Ok(())
    }

    fn visit_i8(&mut self, _depth: u64, val: i8) -> PartialVMResult<()> {
        self.attach(Node::I8(val));
        Ok(())
    }

    fn visit_i16(&mut self, _depth: u64, val: i16) -> PartialVMResult<()> {
        self.attach(Node::I16(val));
        Ok(())
    }

    fn visit_i32(&mut self, _depth: u64, val: i32) -> PartialVMResult<()> {
        self.attach(Node::I32(val));
        Ok(())
    }

    fn visit_i64(&mut self, _depth: u64, val: i64) -> PartialVMResult<()> {
        self.attach(Node::I64(val));
        Ok(())
    }

    fn visit_i128(&mut self, _depth: u64, val: i128) -> PartialVMResult<()> {
        self.attach(Node::I128(val));
        Ok(())
    }

    fn visit_i256(&mut self, _depth: u64, val: &I256) -> PartialVMResult<()> {
        self.attach(Node::I256(*val));
        Ok(())
    }

    fn visit_bool(&mut self, _depth: u64, val: bool) -> PartialVMResult<()> {
        self.attach(Node::Bool(val));
        Ok(())
    }

    fn visit_address(&mut self, _depth: u64, val: &AccountAddress) -> PartialVMResult<()> {
        self.attach(Node::Address(*val));
        Ok(())
    }

    fn visit_struct(&mut self, _depth: u64, len: usize) -> PartialVMResult<bool> {
        self.start(Open::Struct, len);
        Ok(true)
    }

    fn visit_closure(
        &mut self,
        _depth: u64,
        fun: &(dyn AbstractFunction + 'static),
        len: usize,
    ) -> PartialVMResult<bool> {
        self.start(Open::Closure(fun.to_canonical_string()), len);
        Ok(true)
    }

    fn visit_vec(&mut self, _depth: u64, len: usize) -> PartialVMResult<bool> {
        self.start(Open::Vector, len);
        Ok(true)
    }

    fn visit_ref(&mut self, _depth: u64, _is_global: bool) -> PartialVMResult<bool> {
        self.start(Open::Ref, 1);
        Ok(true)
    }
}

fn build_tree(view: &impl ValueView) -> Option<Node> {
    let mut builder = TreeBuilder::default();
    view.visit(&mut builder).ok()?;
    builder.done
}

/// The tree of local `idx` of `frame`, or `None` if it holds no value.
fn local_tree(frame: &Frame, idx: usize) -> Option<Node> {
    let mut builder = TreeBuilder::default();
    match frame.locals.visit_loc(idx, &mut builder) {
        Ok(true) => builder.done,
        _ => None,
    }
}

fn strip_ref(node: &Node) -> &Node {
    match node {
        Node::Ref(inner) => strip_ref(inner),
        other => other,
    }
}

// -------------------------------------------------------------------------------------------
// JSON output
// -------------------------------------------------------------------------------------------

fn write_json_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            },
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_struct_tag(out: &mut String, tag: &StructTag) {
    out.push_str("{\"address\":");
    write_json_str(out, &tag.address.to_hex());
    out.push_str(",\"module\":");
    write_json_str(out, tag.module.as_str());
    out.push_str(",\"name\":");
    write_json_str(out, tag.name.as_str());
    out.push_str(",\"type_args\":[");
    for (i, arg) in tag.type_args.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_type_tag(out, arg);
    }
    out.push_str("]}");
}

fn write_type_tag(out: &mut String, tag: &TypeTag) {
    let primitive = match tag {
        TypeTag::Bool => "bool",
        TypeTag::U8 => "u8",
        TypeTag::U16 => "u16",
        TypeTag::U32 => "u32",
        TypeTag::U64 => "u64",
        TypeTag::U128 => "u128",
        TypeTag::U256 => "u256",
        TypeTag::I8 => "i8",
        TypeTag::I16 => "i16",
        TypeTag::I32 => "i32",
        TypeTag::I64 => "i64",
        TypeTag::I128 => "i128",
        TypeTag::I256 => "i256",
        TypeTag::Address => "address",
        TypeTag::Signer => "signer",
        TypeTag::Vector(inner) => {
            out.push_str("{\"vector\":");
            write_type_tag(out, inner);
            out.push('}');
            return;
        },
        TypeTag::Struct(tag) => {
            out.push_str("{\"struct\":");
            write_struct_tag(out, tag);
            out.push('}');
            return;
        },
        other => {
            write_json_str(out, &other.to_canonical_string());
            return;
        },
    };
    write_json_str(out, primitive);
}

fn write_unavailable(out: &mut String) {
    out.push_str(
        "{\"type\":\"Struct\",\"value\":{\"type_\":{\"address\":\"0\",\"module\":\"trace\",\
         \"name\":\"Unavailable\",\"type_args\":[]},\"fields\":[]}}",
    );
}

fn write_struct(
    out: &mut String,
    tag: Option<&StructTag>,
    fallback_name: &str,
    fields: &[(String, &Node, Option<&MoveTypeLayout>)],
) {
    out.push_str("{\"type\":\"Struct\",\"value\":{\"type_\":");
    match tag {
        Some(tag) => write_struct_tag(out, tag),
        None => {
            out.push_str("{\"address\":\"0\",\"module\":\"?\",\"name\":");
            write_json_str(out, fallback_name);
            out.push_str(",\"type_args\":[]}");
        },
    }
    out.push_str(",\"fields\":[");
    write_fields(out, fields);
    out.push_str("]}}");
}

fn write_fields(out: &mut String, fields: &[(String, &Node, Option<&MoveTypeLayout>)]) {
    for (i, (name, node, layout)) in fields.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('[');
        write_json_str(out, name);
        out.push(',');
        write_value(out, node, *layout);
        out.push(']');
    }
}

/// Write `node` as a `SerializableMoveValue`, using `layout` for struct names, field names,
/// variants and element types when it is known.
fn write_value(out: &mut String, node: &Node, layout: Option<&MoveTypeLayout>) {
    // Delayed fields (aggregators) carry their value's layout inside a native layout.
    let layout = match layout {
        Some(MoveTypeLayout::Native(_, inner)) => Some(inner.as_ref()),
        other => other,
    };
    match node {
        Node::U8(v) => {
            let _ = write!(out, "{{\"type\":\"U8\",\"value\":{}}}", v);
        },
        Node::U16(v) => {
            let _ = write!(out, "{{\"type\":\"U16\",\"value\":{}}}", v);
        },
        Node::U32(v) => {
            let _ = write!(out, "{{\"type\":\"U32\",\"value\":{}}}", v);
        },
        Node::U64(v) => {
            let _ = write!(out, "{{\"type\":\"U64\",\"value\":{}}}", v);
        },
        // Wide integers as decimal strings, so that JSON readers do not round them.
        Node::U128(v) => {
            let _ = write!(out, "{{\"type\":\"U128\",\"value\":\"{}\"}}", v);
        },
        Node::U256(v) => {
            let _ = write!(out, "{{\"type\":\"U256\",\"value\":\"{}\"}}", v);
        },
        Node::I8(v) => {
            let _ = write!(out, "{{\"type\":\"I8\",\"value\":{}}}", v);
        },
        Node::I16(v) => {
            let _ = write!(out, "{{\"type\":\"I16\",\"value\":{}}}", v);
        },
        Node::I32(v) => {
            let _ = write!(out, "{{\"type\":\"I32\",\"value\":{}}}", v);
        },
        Node::I64(v) => {
            let _ = write!(out, "{{\"type\":\"I64\",\"value\":{}}}", v);
        },
        Node::I128(v) => {
            let _ = write!(out, "{{\"type\":\"I128\",\"value\":\"{}\"}}", v);
        },
        Node::I256(v) => {
            let _ = write!(out, "{{\"type\":\"I256\",\"value\":\"{}\"}}", v);
        },
        Node::Bool(v) => {
            let _ = write!(out, "{{\"type\":\"Bool\",\"value\":{}}}", v);
        },
        Node::Address(a) => {
            out.push_str("{\"type\":\"Address\",\"value\":");
            write_json_str(out, &a.to_hex());
            out.push('}');
        },
        Node::Ref(inner) => write_value(out, inner, layout),
        Node::Delayed(id) => {
            let _ = write!(
                out,
                "{{\"type\":\"Struct\",\"value\":{{\"type_\":{{\"address\":\"1\",\"module\":\"aggregator_v2\",\
                 \"name\":\"DelayedFieldID\",\"type_args\":[]}},\"fields\":[[\"id\",{{\"type\":\"U64\",\"value\":{}}}]]}}}}",
                id
            );
        },
        Node::Closure(name, captured) => {
            let fields: Vec<_> = captured
                .iter()
                .enumerate()
                .map(|(i, n)| (format!("captured{}", i), n, None))
                .collect();
            write_struct(out, None, name, &fields);
        },
        Node::Vector(elems) => {
            let elem_layout = match layout {
                Some(MoveTypeLayout::Vector(inner)) => Some(inner.as_ref()),
                _ => None,
            };
            out.push_str("{\"type\":\"Vector\",\"value\":[");
            for (i, elem) in elems.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(out, elem, elem_layout);
            }
            out.push_str("]}");
        },
        Node::Struct(kids) => match layout {
            // A signer is a struct holding its address (and, for permissioned signers, more).
            Some(MoveTypeLayout::Signer) => {
                match kids.iter().rev().find(|k| matches!(k, Node::Address(_))) {
                    Some(addr) => write_value(out, addr, None),
                    None => write_value(out, node, None),
                }
            },
            Some(MoveTypeLayout::Struct(struct_layout)) => {
                write_struct_with_layout(out, kids, struct_layout)
            },
            _ => {
                let fields: Vec<_> = kids
                    .iter()
                    .enumerate()
                    .map(|(i, n)| (format!("field{}", i), n, None))
                    .collect();
                write_struct(out, None, "?", &fields);
            },
        },
    }
}

fn write_struct_with_layout(out: &mut String, kids: &[Node], layout: &MoveStructLayout) {
    match layout {
        // With enum-based options, an `Option` value is an enum (its tag, then the value if
        // any) while its annotated layout is the legacy `{ vec: vector<T> }` struct: render it
        // in the legacy shape.
        MoveStructLayout::WithTypes { type_, fields }
            if type_.is_option() && matches!(kids.first(), Some(Node::U16(_))) =>
        {
            let elem_layout = match fields.first().map(|f| &f.layout) {
                Some(MoveTypeLayout::Vector(inner)) => Some(inner.as_ref()),
                _ => None,
            };
            out.push_str("{\"type\":\"Struct\",\"value\":{\"type_\":");
            write_struct_tag(out, type_);
            out.push_str(",\"fields\":[[\"vec\",{\"type\":\"Vector\",\"value\":[");
            for (i, kid) in kids.iter().skip(1).enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(out, kid, elem_layout);
            }
            out.push_str("]}]]}}");
        },
        MoveStructLayout::WithTypes { type_, fields } => {
            let fields: Vec<_> = kids
                .iter()
                .enumerate()
                .map(|(i, n)| match fields.get(i) {
                    Some(f) => (f.name.to_string(), n, Some(&f.layout)),
                    None => (format!("field{}", i), n, None),
                })
                .collect();
            write_struct(out, Some(type_), "?", &fields);
        },
        MoveStructLayout::WithFields(fields) => {
            let fields: Vec<_> = kids
                .iter()
                .enumerate()
                .map(|(i, n)| match fields.get(i) {
                    Some(f) => (f.name.to_string(), n, Some(&f.layout)),
                    None => (format!("field{}", i), n, None),
                })
                .collect();
            write_struct(out, None, "?", &fields);
        },
        MoveStructLayout::Runtime(field_layouts) => {
            let fields: Vec<_> = kids
                .iter()
                .enumerate()
                .map(|(i, n)| (format!("field{}", i), n, field_layouts.get(i)))
                .collect();
            write_struct(out, None, "?", &fields);
        },
        MoveStructLayout::WithVariants { type_, variants } => {
            // An enum value is a struct whose first field is the variant tag.
            let tag = match kids.first() {
                Some(Node::U16(t)) => *t,
                _ => 0,
            };
            let variant = variants.get(tag as usize);
            let fields: Vec<_> = kids
                .iter()
                .skip(1)
                .enumerate()
                .map(|(i, n)| match variant.and_then(|v| v.fields.get(i)) {
                    Some(f) => (f.name.to_string(), n, Some(&f.layout)),
                    None => (format!("pos{}", i), n, None),
                })
                .collect();
            out.push_str("{\"type\":\"Variant\",\"value\":{\"type_\":");
            write_struct_tag(out, type_);
            out.push_str(",\"variant_name\":");
            match variant {
                Some(v) => write_json_str(out, v.name.as_str()),
                None => out.push_str("null"),
            }
            let _ = write!(out, ",\"tag\":{},\"fields\":[", tag);
            write_fields(out, &fields);
            out.push_str("]}}");
        },
        MoveStructLayout::RuntimeVariants(variants) => {
            let tag = match kids.first() {
                Some(Node::U16(t)) => *t,
                _ => 0,
            };
            let variant = variants.get(tag as usize);
            let fields: Vec<_> = kids
                .iter()
                .skip(1)
                .enumerate()
                .map(|(i, n)| (format!("pos{}", i), n, variant.and_then(|v| v.get(i))))
                .collect();
            let _ = write!(
                out,
                "{{\"type\":\"Variant\",\"value\":{{\"type_\":{{\"address\":\"0\",\"module\":\"?\",\"name\":\"?\",\"type_args\":[]}},\"variant_name\":null,\"tag\":{},\"fields\":[",
                tag
            );
            write_fields(out, &fields);
            out.push_str("]}}");
        },
    }
}
