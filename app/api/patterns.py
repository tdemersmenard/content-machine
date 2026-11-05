"""Routes de la bibliothèque de patterns."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.models import Pattern, PatternType
from app.db.session import get_db
from app.patterns.embeddings import Embedder, embed_texts
from app.patterns.schemas import PatternDetail, PatternOut, PatternSearchHit
from app.patterns.search import search_patterns, to_detail, to_out, top_patterns

router = APIRouter(prefix="/patterns", tags=["patterns"])


def get_embedder() -> Embedder:
    return embed_texts


@router.get("", response_model=list[PatternOut])
def list_patterns(
    db: Annotated[Session, Depends(get_db)],
    type: PatternType | None = None,
    niche_id: int | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[PatternOut]:
    try:
        return [to_out(item) for item in top_patterns(db, type, niche_id, limit)]
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


# Déclarée avant /{pattern_id}, sinon « search » serait lu comme un id.
@router.get("/search", response_model=list[PatternSearchHit])
def search(
    db: Annotated[Session, Depends(get_db)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    q: Annotated[str, Query(min_length=1)],
    type: PatternType | None = None,
    k: Annotated[int, Query(ge=1, le=50)] = 10,
) -> list[PatternSearchHit]:
    return search_patterns(db, q, type, k, embedder)


@router.get("/{pattern_id}", response_model=PatternDetail)
def get_pattern(pattern_id: int, db: Annotated[Session, Depends(get_db)]) -> PatternDetail:
    pattern = db.get(Pattern, pattern_id)
    if pattern is None:
        raise HTTPException(404, "Pattern inconnu.")
    return to_detail(db, pattern)
