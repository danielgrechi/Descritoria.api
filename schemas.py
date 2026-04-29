from __future__ import annotations
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field

from models import TipoDescricao, TipoFeedback, TipoTransacao


# ── Autenticação ──────────────────────────────────────────────────────────────

class RegistroRequest(BaseModel):
    nome: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    senha: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    email: EmailStr
    senha: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenRefreshRequest(BaseModel):
    token: str


# ── Usuário ───────────────────────────────────────────────────────────────────

class PerfilResponse(BaseModel):
    id: int
    nome: str
    email: str
    config_voz: str
    created_at: datetime

    class Config:
        from_attributes = True


class PerfilUpdateRequest(BaseModel):
    nome: Optional[str] = Field(None, min_length=2, max_length=100)
    config_voz: Optional[str] = None


class SaldoResponse(BaseModel):
    saldo: float
    moeda: str = "PerceptMoney"


class ConfiguracoesResponse(BaseModel):
    config_voz: str
    integracoes: dict


class ConfiguracoesUpdateRequest(BaseModel):
    config_voz: Optional[str] = None


class EstiloDescricaoResponse(BaseModel):
    estilo: str
    perfil_aprendizado_ia: str
    preferencias_extraidas: dict
    ultima_atualizacao_perfil: Optional[datetime]
    ativo: bool


class EstiloDescricaoUpdateRequest(BaseModel):
    estilo: str = Field("", max_length=2000)


class PerfilAprendizadoUpdateRequest(BaseModel):
    perfil_aprendizado_ia: str = Field("", max_length=4000)


class CartaoRequest(BaseModel):
    numero: str = Field(..., min_length=16, max_length=16)
    validade: str = Field(..., pattern=r"^\d{2}/\d{2}$")
    cvv: str = Field(..., min_length=3, max_length=4)
    nome_titular: str


class CartaoResponse(BaseModel):
    id: int
    ultimos4: str
    bandeira: str
    principal: bool

    class Config:
        from_attributes = True


# ── Imagens ───────────────────────────────────────────────────────────────────

class DescricaoResponse(BaseModel):
    id: int
    descricoes: List[str]
    modelo: str
    formato_original: str
    tipo: TipoDescricao
    created_at: datetime

    class Config:
        from_attributes = True


class PerguntaRequest(BaseModel):
    descricao_id: int
    pergunta: str = Field(..., min_length=1, max_length=500)


class PerguntaNovaRequest(BaseModel):
    pergunta: str = Field(..., min_length=1, max_length=500)


class RespostaRequest(BaseModel):
    resposta: str
    modelo: str


class HistoricoItemResponse(BaseModel):
    id: int
    tipo: TipoDescricao
    descricao: str
    modelo: str
    formato_original: Optional[str]
    salvo: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ── Estabelecimentos ──────────────────────────────────────────────────────────

class EstabelecimentoCreate(BaseModel):
    nome: str = Field(..., min_length=2, max_length=200)
    categoria: str
    endereco: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    descricao: Optional[str] = None
    telefone: Optional[str] = None
    site: Optional[str] = None


class EstabelecimentoUpdate(BaseModel):
    nome: Optional[str] = None
    categoria: Optional[str] = None
    endereco: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    descricao: Optional[str] = None
    telefone: Optional[str] = None
    site: Optional[str] = None


class EstabelecimentoResponse(BaseModel):
    id: int
    nome: str
    categoria: str
    endereco: Optional[str]
    lat: Optional[float]
    lng: Optional[float]
    descricao: Optional[str]
    telefone: Optional[str]
    site: Optional[str]
    verificado: bool
    avaliacao_media: float
    total_avaliacoes: int

    class Config:
        from_attributes = True


class AvaliacaoRequest(BaseModel):
    nota: int = Field(..., ge=1, le=5)
    comentario: Optional[str] = None


# ── Skills ────────────────────────────────────────────────────────────────────

class SkillCreate(BaseModel):
    nome: str = Field(..., min_length=2, max_length=100)
    descricao: Optional[str] = None
    preco: float = Field(0.0, ge=0.0)
    categoria: str
    prompt_sistema: str = Field(..., min_length=10)
    config_json: Optional[str] = "{}"
    publicado: bool = False


class SkillResponse(BaseModel):
    id: int
    nome: str
    descricao: Optional[str]
    preco: float
    categoria: str
    criador_id: int
    publicado: bool
    avaliacao_media: float
    total_assinantes: int
    em_destaque: bool
    em_promocao: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ── Feedback ──────────────────────────────────────────────────────────────────

class FeedbackBomRequest(BaseModel):
    descricao_id: int
    comentario: Optional[str] = Field(None, max_length=500)


class FeedbackRuimRequest(BaseModel):
    descricao_id: int
    comentario: str = Field(..., min_length=5, max_length=1000)


class FeedbackResponse(BaseModel):
    id: int
    tipo: TipoFeedback
    descricao_id: int
    created_at: datetime

    class Config:
        from_attributes = True


# ── Transações ────────────────────────────────────────────────────────────────

class TransacaoResponse(BaseModel):
    id: int
    tipo: TipoTransacao
    valor: float
    descricao: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True
