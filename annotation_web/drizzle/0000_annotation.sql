CREATE TABLE users (id TEXT PRIMARY KEY NOT NULL, email TEXT NOT NULL UNIQUE, name TEXT NOT NULL, role TEXT NOT NULL);
--> statement-breakpoint
CREATE TABLE invites (email TEXT PRIMARY KEY NOT NULL, name TEXT NOT NULL);
--> statement-breakpoint
CREATE TABLE tracks (id TEXT PRIMARY KEY NOT NULL, title TEXT NOT NULL, original TEXT NOT NULL, edits TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', assignee TEXT, version INTEGER NOT NULL DEFAULT 0, audio_key TEXT, audio_size INTEGER, created TEXT NOT NULL, updated TEXT NOT NULL);
--> statement-breakpoint
CREATE TABLE events (id TEXT PRIMARY KEY NOT NULL, track_id TEXT NOT NULL, user_id TEXT NOT NULL, action TEXT NOT NULL, at TEXT NOT NULL, details TEXT NOT NULL);
--> statement-breakpoint
CREATE INDEX events_track ON events(track_id, at);
--> statement-breakpoint
CREATE INDEX tracks_assignee ON tracks(assignee, status);
