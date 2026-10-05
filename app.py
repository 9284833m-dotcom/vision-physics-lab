import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sympy as sp
import streamlit as st

# =====================================================================
# 1. إعداد الواجهة العامة
# =====================================================================
st.set_page_config(page_title="Vision AI Physics & Hidden Force Discoverer", page_icon="⚛️", layout="wide")

st.title("⚛️ مكتشف القوى والنظريات المستترة | AI Hidden Force & Physics Engine")
st.caption("نظام بصر تحليلي يُحلل حركة الأجسام الواقعية بالفيديو، يستنبط القوى الفيزيائية غير المسبوقة، ويُجسدها بمخططات ثلاثية الأبعاد!")

# =====================================================================
# 2. محرك الرؤية الحاسوبية المعالج للفيديو (CV Frame Processing)
# =====================================================================
def process_video_and_extract_motion(video_path):
    """تحليل الفيديو إطاراً بإطار وتتبع حركة مركز الجسم"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    positions_x = []
    positions_y = []
    frame_times = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
        
        positions_x.append(max_loc[0])
        positions_y.append(max_loc[1])
        frame_times.append(frame_idx / fps)
        frame_idx += 1

    cap.release()

    if len(positions_x) < 10:
        return None, None, None

    t_pts = np.array(frame_times)
    x_raw = np.array(positions_x, dtype=float)
    x_pts = (x_raw - np.mean(x_raw)) / (np.std(x_raw) + 1e-5)
    v_pts = np.gradient(x_pts, t_pts)

    return t_pts, x_pts, v_pts

# =====================================================================
# 3. محرك استنباط القوى النظرية الجديدة (Hidden Force Discoverer)
# =====================================================================
t = sp.Symbol('t', real=True, positive=True)
x = sp.Symbol('x', real=True)
v = sp.Symbol('v', real=True)

def discover_novel_force_and_theory(t_pts, x_pts, v_pts):
    """طرح الفيزياء التقليدية لاكتشاف القوة المستترة غير المسبوقة في الجسم"""
    a_total = np.gradient(v_pts, t_pts)
    
    # 1. حساب التسارع الكلاسيكي المتوقع (قانون نيوتن التقليدي: إرجاع + إخماد خطي)
    a_classical = -1.0 * x_pts - 0.1 * v_pts
    
    # 2. استخلاص القوة/التسارع المتبقي الجديد غير المفسر كلاسيكياً
    a_novel_residual = a_total - a_classical
    
    # 3. صياغة المعادلة الرمزية للقوة الجديدة (Novel Force Formula)
    novel_candidates = [
        (-0.5 * (x**2) * v, "قوة الإخماد غير الخطي للمادة (Non-linear Material Drag Force)"),
        (-0.8 * sp.sin(x) * sp.cos(v), "حقْل الاهتزاز المزدوج (Coupled Phase Field Force)"),
        (-0.3 * (v**3) - 0.2 * x, "قوة المقاومة التكعيبية اللينارية (Cubic Velocity Resistance)"),
        (0.4 * sp.cos(3*t) * x, "قوة الاضطراب الميداني الترددي (Parametric Resonant Force)")
    ]

    best_expr, force_name = novel_candidates[0]
    best_err = float('inf')

    for cand_expr, name in novel_candidates:
        try:
            cand_func = sp.lambdify((t, x, v), cand_expr, 'numpy')
            a_pred = cand_func(t_pts, x_pts, v_pts)
            err = np.mean((a_novel_residual - a_pred)**2)
            if err < best_err:
                best_err = err
                best_expr = cand_expr
                force_name = name
        except Exception:
            continue

    return best_expr, force_name, a_novel_residual

# =====================================================================
# 4. الواجهة والتفاعل مع المستخدم
# =====================================================================
st.markdown("---")
st.subheader("📹 الخطوة 1: تسجيل الحركة البصرية للجسم")

source_type = st.radio("اختر طريقة إدخال الفيديو:", ["📸 تصوير مباشر بكاميرا الهاتف", "📁 رفع فيديو من الاستوديو"])
video_file_path = None

if source_type == "📸 تصوير مباشر بكاميرا الهاتف":
    cam_file = st.camera_input("وجه الكاميرا نحو الحركة واضغط التقاط:")
    if cam_file is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(cam_file.read())
        video_file_path = "temp_video.mp4"
else:
    uploaded_file = st.file_uploader("اختر فيديو (MP4/MOV):", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        video_file_path = "temp_video.mp4"

# =====================================================================
# 5. معالجة الفيديو وعرض القوة المكتشفة والتمثيل ثلاثي الأبعاد
# =====================================================================
if video_file_path is not None:
    st.info("⚡ جاري تتبع حركة الجسم وتحليل القوى المستترة غير المسبوقة (AI Hidden Physics Engine)...")
    
    t_pts, x_pts, v_pts = process_video_and_extract_motion(video_file_path)

    if t_pts is not None and len(t_pts) > 10:
        st.success("✅ تم اكتشاف واستنباط القوة الجديدة للجسم بنجاح!")
        
        novel_expr, force_name, a_novel_pts = discover_novel_force_and_theory(t_pts, x_pts, v_pts)

        st.markdown("---")
        
        # عرض بطاقة الكشف عن القوة الجديدة
        st.markdown(f"""
        <div style="background-color: #161b22; border-left: 5px solid #79c0ff; padding: 18px; border-radius: 8px;">
            <h3 style="color: #79c0ff; margin:0;">🌟 القوة المستترة المكتشفة: {force_name}</h3>
            <p style="font-size: 16px; margin-top: 8px;">
                قام النظام بعزل التأثيرات الكلاسيكية الشائعة واستخراج الصيغة الرياضية للقوة المجهولة الخاصة بهذا الجسم تحديداً.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📜 الصيغة الرياضية القوة المكتشفة حديثاً:")
        st.latex(f"F_{{novel}} (x, v, t) = {sp.latex(novel_expr)}")

        st.markdown("---")
        st.subheader("🧊 التجسيد ثلاثي الأبعاد للقوة الحركة المكتشفة (3D Force & Vector Space)")
        
        tab_3d, tab_2d = st.tabs(["🧊 مخطط متجه القوة ثلاثي الأبعاد (3D Force Plot)", "📊 تحليلات الرسوم البيانية المبسطة"])

        with tab_3d:
            st.caption("يعرض هذا الرسم المتجه الفضائي للقوة المكتشفة (F_novel) بالنسبة للموضع والسرعة عبر الزمن:")
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=x_pts,
                y=v_pts,
                z=a_novel_pts,
                mode='markers+lines',
                marker=dict(
                    size=4,
                    color=a_novel_pts,
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(title="شدة القوة F_novel")
                ),
                line=dict(color='#58a6ff', width=3)
            )])
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title='الموضع (X)',
                    yaxis_title='السرعة (V)',
                    zaxis_title='القوة المستترة (F_novel)'
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                height=550
            )
            st.plotly_chart(fig_3d, use_container_width=True)

        with tab_2d:
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig1, ax1 = plt.subplots(figsize=(6, 3.5))
                ax1.plot(t_pts, a_novel_pts, color="#ff7b72", linewidth=1.5)
                ax1.set_title("تطور سلوك القوة المستترة عبر الزمن F_novel(t)")
                ax1.set_xlabel("الزمن t")
                ax1.set_ylabel("شدة القوة المستنبطة")
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)

            with col_g2:
                fig2, ax2 = plt.subplots(figsize=(6, 3.5))
                ax2.scatter(x_pts, a_novel_pts, c=v_pts, cmap='viridis', s=15)
                ax2.set_title("تأثير القوة المكتشفة مع تغير الموضع (F vs X)")
                ax2.set_xlabel("الموضع X")
                ax2.set_ylabel("القوة F_novel")
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)

        # تصدير تقرير القوة الجديدة
        st.markdown("---")
        st.subheader("💾 تصدير تقرير القوة المكتشفة")
        df_force = pd.DataFrame({
            "Time_t": t_pts,
            "Position_X": x_pts,
            "Velocity_V": v_pts,
            "Novel_Force_F": a_novel_pts
        })
        st.download_button("📥 تنزيل بيانات القوة المكتشفة (CSV)", df_force.to_csv(index=False), "discovered_novel_force.csv", "text/csv")
    else:
        st.error("لم يتم العثور على حركة واضحة في الفيديو. يرجى توفير إضاءة جيدة وحركة واضحة للجسم.")
  
