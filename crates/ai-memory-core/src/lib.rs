//! Core domain types and errors for ai-memory.
//!
//! This crate is the closure of the project's vocabulary: identifiers, agent
//! kinds, and the workspace-wide error type. Nothing in here performs I/O,
//! which keeps it trivially unit-testable and free of platform concerns.

pub mod error;
pub mod handoff;
pub mod ids;
pub mod observation;
pub mod page;

pub use error::{MemoryError, MemoryResult};
pub use handoff::{Handoff, HandoffState, NewHandoff};
pub use ids::{
    AgentKind, HandoffId, ObservationId, PageId, PagePath, ProjectId, SessionId, WorkspaceId,
};
pub use observation::{NewObservation, NewSession, Observation, ObservationKind};
pub use page::{NewPage, Page, Tier};
