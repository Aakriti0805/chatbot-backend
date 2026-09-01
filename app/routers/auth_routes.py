from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app import models, schemas, auth

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/send-otp")
def send_otp(user: schemas.UserCreate, db: Session = Depends(get_db)):
    otp_code = auth.generate_otp()

    new_otp = models.OTP(email=user.email, otp_code=otp_code)
    db.add(new_otp)
    db.commit()

    auth.send_otp_email(user.email, otp_code)

    return {"message": "OTP sent to your email"}


@router.post("/verify-otp")
def verify_otp(email: str, otp_code: str, db: Session = Depends(get_db)):
    record = (
        db.query(models.OTP)
        .filter(models.OTP.email == email, models.OTP.otp_code == otp_code)
        .order_by(models.OTP.created_at.desc())
        .first()
    )

    if not record:
        raise HTTPException(status_code=400, detail="Invalid OTP")

    existing_user = db.query(models.User).filter(models.User.email == email).first()
    if not existing_user:
        existing_user = models.User(email=email, role="user")
        db.add(existing_user)
        db.commit()
        db.refresh(existing_user)

    access_token = auth.create_access_token(
        data={"sub": existing_user.email, "role": existing_user.role}
    )

    return {"access_token": access_token, "token_type": "bearer", "role": existing_user.role}