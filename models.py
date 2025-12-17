# models.py
from datetime import datetime
from sqlalchemy import Column, Integer, Text, DateTime, ForeignKey, Float, JSON, UniqueConstraint, Boolean, func, BigInteger
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from db import Base

# Note: extend_existing=True prevents SQLAlchemy from crashing if the table already exists.
# We keep column names & types consistent with your SQL.

class AppUser(Base):
    __tablename__ = "app_user"
    __table_args__ = {"extend_existing": True}

    # BIGSERIAL in Postgres → BigInteger in SQLAlchemy
    id = Column(BigInteger, primary_key=True)

    # New columns matching your recreated table
    firebase_uid = Column(Text, unique=True, nullable=True)
    email = Column(Text, unique=True, nullable=True)
    phone = Column(Text, unique=True, nullable=True)
    password_hash = Column(Text, nullable=True)

    # All old columns removed:
    # uid, display_name, photo_url, roles, meta, quota, last_seen, is_active, created_at

    farms = relationship("Farm", back_populates="user")

class Farm(Base):
    # Updated table name to plural to match your DB rename (farm -> farms)
    __tablename__ = "farms"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    user_id = Column(BigInteger, ForeignKey("app_user.id", ondelete="CASCADE"), nullable=True)
    name = Column(Text, nullable=True)
    # geom stored as PostGIS polygon with SRID 4326
    geom = Column(Geometry(geometry_type="POLYGON", srid=4326, spatial_index=True), nullable=False)
    bbox = Column(Geometry(geometry_type="POLYGON", srid=4326), nullable=True)
    area_m2 = Column(Float, nullable=True)
    meta = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    user = relationship("AppUser", back_populates="farms")
    scenes = relationship("Scene", back_populates="farm")
    process_results = relationship("ProcessResult", back_populates="farm")


class Scene(Base):
    __tablename__ = "scene"
    __table_args__ = (UniqueConstraint("farm_id", "scene_id", name="uq_scene_farm_sceneid"), {"extend_existing": True})

    id = Column(Integer, primary_key=True)
    # Updated FK target to point to farms.id (was farm.id)
    farm_id = Column(Integer, ForeignKey("farms.id", ondelete="CASCADE"), nullable=False)
    scene_id = Column(Text, nullable=False)
    collection = Column(Text, nullable=True)
    properties = Column(JSONB, nullable=True)
    score = Column(Float, nullable=True)
    meta = Column(JSONB, nullable=True)
    fetched_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    farm = relationship("Farm", back_populates="scenes")


class ProcessResult(Base):
    __tablename__ = "process_result"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    # Updated FK target to point to farms.id (was farm.id)
    farm_id = Column(Integer, ForeignKey("farms.id", ondelete="CASCADE"), nullable=False)
    source = Column(Text, nullable=True)
    scene_id = Column(Text, nullable=True)
    start_ts = Column(DateTime(timezone=True), nullable=True)
    end_ts = Column(DateTime(timezone=True), nullable=True)
    result = Column(JSONB, nullable=True)    # full returned JSON
    health_score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    farm = relationship("Farm", back_populates="process_results")


class ApiLog(Base):
    __tablename__ = "api_log"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    user_id = Column(BigInteger, ForeignKey("app_user.id"), nullable=True)
    endpoint = Column(Text, nullable=True)
    payload = Column(JSONB, nullable=True)
    response_status = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class UserQuota(Base):
    __tablename__ = "user_quota"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    user_id = Column(BigInteger, ForeignKey("app_user.id"), unique=True, nullable=False)
    monthly_calls = Column(Integer, default=0)
    last_reset = Column(DateTime(timezone=True), default=datetime.utcnow)
