from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, Enum
)
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from database import Base


class TipoFeedback(str, enum.Enum):
    bom = "bom"
    ruim = "ruim"


class TipoDescricao(str, enum.Enum):
    foto = "foto"
    video = "video"
    documento = "documento"


class TipoTransacao(str, enum.Enum):
    credito = "credito"
    debito = "debito"


class Usuario(Base):
    __tablename__ = "usuarios"

    id = Column(Integer, primary_key=True, index=True)
    nome = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    senha_hash = Column(String, nullable=False)
    saldo_perceptmoney = Column(Float, default=0.0)
    config_voz = Column(String, default="padrao")
    integracoes = Column(Text, default="{}")
    estilo_descricao = Column(Text, default="")  # instruções personalizadas do usuário para a IA
    created_at = Column(DateTime, default=datetime.utcnow)

    descricoes = relationship("Descricao", back_populates="usuario")
    feedbacks = relationship("Feedback", back_populates="usuario")
    skills_criadas = relationship("Skill", back_populates="criador")
    assinaturas = relationship("SkillAssinatura", back_populates="usuario")
    transacoes = relationship("Transacao", back_populates="usuario")
    cartoes = relationship("Cartao", back_populates="usuario")
    pessoas_conhecidas = relationship("Pessoa", back_populates="usuario")


class Descricao(Base):
    __tablename__ = "descricoes"

    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    tipo = Column(Enum(TipoDescricao), default=TipoDescricao.foto)
    conteudo_hash = Column(String, index=True)
    descricao = Column(Text, nullable=False)
    modelo = Column(String, nullable=False)
    formato_original = Column(String)
    salvo = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    usuario = relationship("Usuario", back_populates="descricoes")
    feedbacks = relationship("Feedback", back_populates="descricao")


class Feedback(Base):
    __tablename__ = "feedbacks"

    id = Column(Integer, primary_key=True, index=True)
    descricao_id = Column(Integer, ForeignKey("descricoes.id"), nullable=False)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    tipo = Column(Enum(TipoFeedback), nullable=False)
    comentario = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    descricao = relationship("Descricao", back_populates="feedbacks")
    usuario = relationship("Usuario", back_populates="feedbacks")


class Estabelecimento(Base):
    __tablename__ = "estabelecimentos"

    id = Column(Integer, primary_key=True, index=True)
    nome = Column(String, nullable=False)
    categoria = Column(String, nullable=False, index=True)
    endereco = Column(String)
    lat = Column(Float)
    lng = Column(Float)
    descricao = Column(Text)
    telefone = Column(String)
    site = Column(String)
    verificado = Column(Boolean, default=False)
    avaliacao_media = Column(Float, default=0.0)
    total_avaliacoes = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class Skill(Base):
    __tablename__ = "skills"

    id = Column(Integer, primary_key=True, index=True)
    nome = Column(String, nullable=False)
    descricao = Column(Text)
    preco = Column(Float, default=0.0)
    categoria = Column(String, index=True)
    criador_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    prompt_sistema = Column(Text, nullable=False)
    config_json = Column(Text, default="{}")
    publicado = Column(Boolean, default=False)
    avaliacao_media = Column(Float, default=0.0)
    total_assinantes = Column(Integer, default=0)
    em_destaque = Column(Boolean, default=False)
    em_promocao = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    criador = relationship("Usuario", back_populates="skills_criadas")
    assinaturas = relationship("SkillAssinatura", back_populates="skill")


class SkillAssinatura(Base):
    __tablename__ = "skill_assinaturas"

    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    skill_id = Column(Integer, ForeignKey("skills.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    usuario = relationship("Usuario", back_populates="assinaturas")
    skill = relationship("Skill", back_populates="assinaturas")


class Transacao(Base):
    __tablename__ = "transacoes"

    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    tipo = Column(Enum(TipoTransacao), nullable=False)
    valor = Column(Float, nullable=False)
    descricao = Column(String)
    referencia_id = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    usuario = relationship("Usuario", back_populates="transacoes")


class Cartao(Base):
    __tablename__ = "cartoes"

    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    ultimos4 = Column(String(4), nullable=False)
    bandeira = Column(String)
    token_gateway = Column(String)
    principal = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    usuario = relationship("Usuario", back_populates="cartoes")


class Pessoa(Base):
    __tablename__ = "pessoas"

    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    nome = Column(String, nullable=False)
    caracteristicas = Column(Text, nullable=False)  # descrição física armazenada
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    usuario = relationship("Usuario", back_populates="pessoas_conhecidas")
