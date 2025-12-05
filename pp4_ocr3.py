from PIL import Image, ImageEnhance, ImageDraw
import pytesseract
import cv2
import numpy as np
import argparse
import os
import json

# 1. Налаштування Tesseract
# try:
#     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# except:
#     pass

# Константи
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
CONFIG_RUS = '-l rus --psm 6'
CONFIG_FILE = 'roi_coordinates.json'
PREPROCESSING_CONFIG_FILE = 'preprocessing_settings.json'

# Глобальні змінні
current_image = None
click_points = []
roi_data = {}

# --- Функції для роботи з конфігурацією ---

def save_coordinates(coords, filename=CONFIG_FILE):
    """Зберігає координати ROI у JSON"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(coords, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Координати збережено: {filename}")

def load_coordinates(filename=CONFIG_FILE):
    """Завантажує координати ROI"""
    if not os.path.exists(filename):
        return None
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_preprocessing_settings(settings, filename=PREPROCESSING_CONFIG_FILE):
    """Зберігає налаштування препроцесингу"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)
    print(f"✓ Налаштування обробки збережено: {filename}")

def load_preprocessing_settings(filename=PREPROCESSING_CONFIG_FILE):
    """Завантажує налаштування препроцесингу"""
    if not os.path.exists(filename):
        return get_default_preprocessing_settings()
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_default_preprocessing_settings():
    """Налаштування обробки за замовчуванням"""
    return {
        'invert': True,
        'threshold_method': 'otsu',  # 'otsu', 'adaptive', 'binary'
        'threshold_value': 127,
        'blur_size': 3,
        'contrast': 1.5,
        'sharpness': 1.3,
        'denoise': True,
        'morph_operations': False,
        'scale_factor': 1.0
    }

# --- Меню ---

def setup_main_menu():
    """Головне меню налаштування"""
    print("\n" + "="*70)
    print("МЕНЮ НАЛАШТУВАННЯ")
    print("="*70)
    print("1. Повне налаштування з нуля (всі елементи)")
    print("2. Редагувати конкретний тип полів")
    print("3. Редагувати конкретне замовлення")
    print("0. Вихід")
    
    choice = input("\nВиберіть опцію: ").strip()
    return choice

def field_type_menu():
    """Меню вибору типу поля"""
    print("\n" + "="*70)
    print("ВИБІР ТИПУ ПОЛЯ")
    print("="*70)
    print("1. Заголовок (Кафе)")
    print("2. Водойми (список зліва, 17 шт)")
    print("3. Глобальний таймер")
    print("4. Картки замовлень (основні рамки)")
    print("5. Назви риб на всіх картках")
    print("6. Таймери на всіх картках")
    print("7. Кількість на всіх картках")
    print("8. Вага на всіх картках")
    print("9. Ціна на всіх картках")
    print("0. Назад")
    
    choice = input("\nВиберіть тип поля: ").strip()
    return choice

def order_menu():
    """Меню вибору замовлення"""
    print("\n" + "="*70)
    print("ВИБІР ЗАМОВЛЕННЯ")
    print("="*70)
    for i in range(8):
        print(f"{i+1}. Замовлення {i+1}")
    print("0. Назад")
    
    choice = input("\nВиберіть замовлення: ").strip()
    return choice

def order_field_menu():
    """Меню вибору поля в замовленні"""
    print("\n" + "="*70)
    print("ВИБІР ПОЛЯ")
    print("="*70)
    print("1. Основна рамка картки")
    print("2. Назва риби")
    print("3. Таймер")
    print("4. Кількість")
    print("5. Вага")
    print("6. Ціна")
    print("0. Назад")
    
    choice = input("\nВиберіть поле: ").strip()
    return choice

def debug_menu():
    """Меню вибору для debug режиму"""
    print("\n" + "="*70)
    print("DEBUG РЕЖИМ - РЕДАГУВАННЯ")
    print("="*70)
    print("1. Редагувати тип полів")
    print("2. Редагувати конкретне замовлення")
    print("0. Вихід без змін")
    
    choice = input("\nВиберіть опцію: ").strip()
    return choice

# --- Інтерактивний вибір ROI ---

def mouse_callback(event, x, y, flags, param):
    """Обробник кліків миші"""
    global click_points, current_image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((x, y))
        
        img_copy = current_image.copy()
        for i, pt in enumerate(click_points):
            cv2.circle(img_copy, pt, 5, (0, 255, 0), -1)
            cv2.putText(img_copy, str(i+1), (pt[0]+10, pt[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if len(click_points) == 2:
            cv2.rectangle(img_copy, click_points[0], click_points[1], (0, 255, 0), 2)
        
        cv2.imshow('ROI Selection', img_copy)

def select_roi(image, instruction):
    """Вибір області ROI"""
    global click_points, current_image
    
    click_points = []
    current_image = image.copy()
    
    print(f"\n{instruction}")
    print("ЛКМ: верхній лівий → нижній правий | ENTER: підтвердити | ESC: пропустити")
    
    cv2.namedWindow('ROI Selection')
    cv2.setMouseCallback('ROI Selection', mouse_callback)
    cv2.imshow('ROI Selection', current_image)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter
            if len(click_points) == 2:
                x1, y1 = click_points[0]
                x2, y2 = click_points[1]
                roi = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                cv2.destroyWindow('ROI Selection')
                return roi
            else:
                print("⚠ Потрібно вибрати 2 точки!")
        
        elif key == 27:  # ESC
            cv2.destroyWindow('ROI Selection')
            return None

def interactive_preprocessing_setup(img_path, coords, field_type='price', order_num=None):
    """
    Інтерактивне налаштування обробки з можливістю перемикання між замовленнями
    field_type: 'price', 'fish_name', 'quantity', 'weight', 'timer', 'header', 'global_timer'
    order_num: якщо вказано, налаштовуємо тільки це замовлення
    """
    print("\n" + "="*70)
    print("НАЛАШТУВАННЯ ОБРОБКИ ЗОБРАЖЕННЯ")
    print("="*70)
    
    pil_img = normalize_image(img_path, TARGET_WIDTH, TARGET_HEIGHT)
    settings = load_preprocessing_settings()  # Завантажуємо існуючі налаштування
    
    # Знаходимо всі доступні області для цього типу поля
    available_orders = []
    
    # Для полів що не в замовленнях
    if field_type == 'header' and coords.get('header'):
        available_orders.append({'index': -1, 'roi': coords['header'], 'name': 'Заголовок (Кафе)'})
    elif field_type == 'global_timer' and coords.get('global_timer'):
        available_orders.append({'index': -1, 'roi': coords['global_timer'], 'name': 'Глобальний таймер'})
    else:
        # Для полів в замовленнях
        for i, order in enumerate(coords.get('orders', [])):
            # Якщо вказано конкретне замовлення, берем тільки його
            if order_num is not None and i != (order_num - 1):
                continue
                
            roi = None
            
            if field_type == 'price' and 'price' in order:
                roi = order['price']
            elif field_type == 'quantity' and 'quantity' in order:
                roi = order['quantity']
            elif field_type == 'weight' and 'weight' in order:
                roi = order['weight']
            elif field_type == 'timer' and 'timer' in order:
                roi = order['timer']
            elif field_type == 'fish_name' and i < len(coords.get('fish_names', [])):
                roi = coords['fish_names'][i]
            elif field_type == 'card' and 'full_box' in order:
                roi = order['full_box']
            
            if roi:
                available_orders.append({'index': i, 'roi': roi, 'name': f"Замовлення {i+1}"})
    
    if not available_orders:
        print(f"⚠ Не знайдено областей типу '{field_type}'")
        return None
    
    current_order_idx = 0
    
    def update_preview():
        order_info = available_orders[current_order_idx]
        roi = order_info['roi']
        test_img = pil_img.crop(roi)
        processed = apply_preprocessing(test_img, settings)
        
        # Збільшуємо для зручності
        scale = 3
        cv_img = cv2.cvtColor(np.array(processed), cv2.COLOR_GRAY2BGR)
        cv_img = cv2.resize(cv_img, (cv_img.shape[1]*scale, cv_img.shape[0]*scale))
        
        # Показуємо поточні налаштування
        info_height = 350
        info_panel = np.zeros((info_height, max(cv_img.shape[1], 600), 3), dtype=np.uint8)
        
        y_offset = 25
        cv2.putText(info_panel, f"Поле: {field_type.upper()}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 30
        cv2.putText(info_panel, f"{order_info['name']} [{current_order_idx+1}/{len(available_orders)}]", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 30
        
        cv2.putText(info_panel, "Klawishi: I-invert T-threshold B-blur C-contrast", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_offset += 20
        cv2.putText(info_panel, "S-sharp N-denoise M-morph A/D-navigate", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_offset += 25
        
        for key, value in settings.items():
            text = f"{key}: {value}"
            color = (0, 255, 0)
            cv2.putText(info_panel, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.45, color, 1)
            y_offset += 22
        
        # Обрізаємо панель до потрібної ширини
        info_panel = info_panel[:, :cv_img.shape[1], :]
        
        # Об'єднуємо панель і зображення
        combined = np.vstack([info_panel, cv_img])
        cv2.imshow('Preprocessing Setup', combined)
    
    print(f"\nЗнайдено {len(available_orders)} областей з полем '{field_type}'")
    print("\nКлавіші керування:")
    print("  A/D - перемикання між замовленнями")
    print("  I - інверсія (так/ні)")
    print("  T - метод threshold (otsu/adaptive/binary)")
    print("  +/- - threshold value")
    print("  B - розмір blur (1/3/5)")
    print("  C - контраст (+/- 0.1)")
    print("  S - різкість (+/- 0.1)")
    print("  N - denoise (так/ні)")
    print("  M - morph operations (так/ні)")
    print("  ENTER - зберегти, ESC - скасувати")
    
    update_preview()
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        changed = False
        
        # Перемикання між замовленнями (тільки якщо більше 1)
        if len(available_orders) > 1:
            if key == ord('d') or key == ord('D'):  # D - вправо
                current_order_idx = (current_order_idx + 1) % len(available_orders)
                print(f"\n→ {available_orders[current_order_idx]['name']}")
                changed = True
            elif key == ord('a') or key == ord('A'):  # A - вліво
                current_order_idx = (current_order_idx - 1) % len(available_orders)
                print(f"\n← {available_orders[current_order_idx]['name']}")
                changed = True
        
        # Налаштування
        if key == ord('i') or key == ord('I'):
            settings['invert'] = not settings['invert']
            print(f"Інверсія: {settings['invert']}")
            changed = True
            
        elif key == ord('t') or key == ord('T'):
            methods = ['otsu', 'adaptive', 'binary']
            idx = methods.index(settings['threshold_method'])
            settings['threshold_method'] = methods[(idx + 1) % len(methods)]
            print(f"Threshold метод: {settings['threshold_method']}")
            changed = True
            
        elif key == ord('+') or key == ord('='):
            settings['threshold_value'] = min(255, settings['threshold_value'] + 10)
            print(f"Threshold value: {settings['threshold_value']}")
            changed = True
            
        elif key == ord('-') or key == ord('_'):
            settings['threshold_value'] = max(0, settings['threshold_value'] - 10)
            print(f"Threshold value: {settings['threshold_value']}")
            changed = True
            
        elif key == ord('b') or key == ord('B'):
            sizes = [1, 3, 5]
            idx = sizes.index(settings['blur_size'])
            settings['blur_size'] = sizes[(idx + 1) % len(sizes)]
            print(f"Blur size: {settings['blur_size']}")
            changed = True
            
        elif key == ord('c') or key == ord('C'):
            settings['contrast'] = round(settings['contrast'] + 0.1, 1)
            if settings['contrast'] > 2.0:
                settings['contrast'] = 0.5
            print(f"Contrast: {settings['contrast']}")
            changed = True
            
        elif key == ord('s') or key == ord('S'):
            settings['sharpness'] = round(settings['sharpness'] + 0.1, 1)
            if settings['sharpness'] > 2.0:
                settings['sharpness'] = 0.5
            print(f"Sharpness: {settings['sharpness']}")
            changed = True
            
        elif key == ord('n') or key == ord('N'):  # Змінено з D на N для denoise
            settings['denoise'] = not settings['denoise']
            print(f"Denoise: {settings['denoise']}")
            changed = True
            
        elif key == ord('m') or key == ord('M'):
            settings['morph_operations'] = not settings['morph_operations']
            print(f"Morph operations: {settings['morph_operations']}")
            changed = True
            
        elif key == 13:  # Enter
            cv2.destroyAllWindows()
            print("\n✓ Налаштування збережено")
            return settings
            
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            print("\n✗ Скасовано")
            return None
        
        if changed:
            update_preview()

# --- Функції пост-обробки розпізнаного тексту ---

def clean_fish_name(text):
    """Очищає назву риби - залишає тільки букви та пробіли"""
    import re
    # Видаляємо все крім кирилиці, латиниці та пробілів
    cleaned = re.sub(r'[^а-яА-ЯёЁa-zA-Z\s]', '', text)
    # Видаляємо зайві пробіли
    cleaned = ' '.join(cleaned.split())
    return cleaned.strip()

def clean_timer(text):
    """Очищає таймер - залишає тільки цифри та форматує як MM:SS"""
    import re
    # Витягуємо всі цифри
    digits = re.sub(r'\D', '', text)
    
    if len(digits) >= 3:
        # Беремо останні 4 цифри або всі що є
        if len(digits) >= 4:
            digits = digits[-4:]
        # Додаємо двокрапку перед останніми 2 цифрами
        minutes = digits[:-2] if len(digits) > 2 else '0'
        seconds = digits[-2:]
        return f"{minutes}:{seconds}"
    elif len(digits) == 2:
        return f"0:{digits}"
    elif len(digits) == 1:
        return f"0:0{digits}"
    
    return text  # Якщо нічого не знайшли, повертаємо оригінал

def clean_quantity(text):
    """Очищує кількість - витягує тільки число"""
    import re
    # Шукаємо перше число в тексті
    match = re.search(r'\d+', text)
    if match:
        return match.group(0)
    return text

def clean_weight(text):
    """Очищує вагу - залишає цифри, кому/крапку та одиниці виміру"""
    import re
    # Видаляємо все крім цифр, коми, крапки, букв к, г, К, Г
    cleaned = re.sub(r'[^0-9.,кгКГ\s]', '', text)
    
    # Шукаємо число з комою або крапкою
    weight_match = re.search(r'(\d+[.,]?\d*)\s*(кг|г|КГ|Г)?', cleaned, re.IGNORECASE)
    if weight_match:
        number = weight_match.group(1).replace('.', ',')
        unit = weight_match.group(2) if weight_match.group(2) else ''
        unit = unit.lower() if unit else ''
        return f"{number} {unit}".strip()
    
    return text

def clean_price(text):
    """Очищує ціну - додає кому перед останніми 2 цифрами якщо її немає"""
    import re
    # Видаляємо все крім цифр, коми та крапки
    cleaned = re.sub(r'[^0-9.,]', '', text)
    
    # Якщо вже є кома або крапка, замінюємо крапку на кому
    if ',' in cleaned or '.' in cleaned:
        return cleaned.replace('.', ',')
    
    # Якщо немає коми/крапки, додаємо кому перед останніми 2 цифрами
    if len(cleaned) >= 3:
        return f"{cleaned[:-2]},{cleaned[-2:]}"
    
    return cleaned

# --- Функції обробки ---

def normalize_image(img_path, target_w, target_h):
    """Масштабує до 1920x1080"""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Файл не знайдено: {img_path}")
    
    img = Image.open(img_path)
    original_size = img.size
    print(f"Оригінальний розмір: {original_size[0]}x{original_size[1]}")
    
    img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    print(f"Уніфіковано до: {target_w}x{target_h}")
    
    return img_resized

def apply_preprocessing(pil_img, settings):
    """Застосовує налаштування обробки"""
    # Контраст
    if settings.get('contrast', 1.0) != 1.0:
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(settings['contrast'])
    
    # Різкість
    if settings.get('sharpness', 1.0) != 1.0:
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(settings['sharpness'])
    
    # Grayscale
    img_gray = pil_img.convert('L')
    img_array = np.array(img_gray)
    
    # Масштабування
    if settings.get('scale_factor', 1.0) != 1.0:
        scale = settings['scale_factor']
        new_size = (int(img_array.shape[1] * scale), int(img_array.shape[0] * scale))
        img_array = cv2.resize(img_array, new_size)
    
    # Інверсія
    if settings.get('invert', False):
        img_array = cv2.bitwise_not(img_array)
    
    # Threshold
    method = settings.get('threshold_method', 'otsu')
    if method == 'otsu':
        _, img_thresh = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        img_thresh = cv2.adaptiveThreshold(img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
    else:  # binary
        _, img_thresh = cv2.threshold(img_array, settings.get('threshold_value', 127), 
                                     255, cv2.THRESH_BINARY)
    
    # Denoise
    if settings.get('denoise', False):
        blur_size = settings.get('blur_size', 3)
        img_thresh = cv2.medianBlur(img_thresh, blur_size)
    
    # Морфологічні операції
    if settings.get('morph_operations', False):
        kernel = np.ones((2, 2), np.uint8)
        img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
    
    return Image.fromarray(img_thresh)

def detect_active_water_body(pil_img, water_coords):
    """
    Визначає активну водойму, порівнюючи середній колір фону.
    Активна водойма зазвичай має інший фон.
    """
    if not water_coords:
        return "Не визначено"

    avg_colors = {}
    for name, roi in water_coords.items():
        water_img = pil_img.crop(roi)
        img_array = np.array(water_img)
        avg_colors[name] = np.mean(img_array, axis=(0, 1))

    # Знаходимо найбільш унікальний колір
    color_distances = {}
    for name1, color1 in avg_colors.items():
        total_distance = 0
        for name2, color2 in avg_colors.items():
            if name1 != name2:
                total_distance += np.linalg.norm(color1 - color2)
        color_distances[name1] = total_distance

    if not color_distances:
        return "Не визначено"
        
    # Водойма з максимальною відстанню від інших є активною
    active_water = max(color_distances, key=color_distances.get)
    return active_water

def is_card_filled(card_img, threshold=0.05):
    """
    Перевіряє заповненість картки за допомогою аналізу щільності країв.
    Поріг (threshold) - це відсоток пікселів, які мають бути краями.
    """
    img_gray = np.array(card_img.convert('L'))

    # Використовуємо Canny edge detection
    edges = cv2.Canny(img_gray, 50, 150)

    # Розраховуємо щільність країв
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

    return edge_density > threshold

def process_with_coordinates(img_path, coords, preproc_settings):
    """Обробка з координатами"""
    
    pil_img = normalize_image(img_path, TARGET_WIDTH, TARGET_HEIGHT)
    
    results = {
        'header': '',
        'active_location': '',
        'timer': '',
        'orders': []
    }
    
    # Заголовок
    if coords.get('header'):
        header_img = pil_img.crop(coords['header'])
        header_img = apply_preprocessing(header_img, preproc_settings)
        results['header'] = pytesseract.image_to_string(header_img, config=CONFIG_RUS).strip()
        print(f"\n✓ Заголовок: {results['header']}")
    
    # Активна водойма
    water_bodies = list(coords.get('water_bodies', {}).keys())
    if water_bodies:
        active_water = detect_active_water_body(pil_img, coords['water_bodies'])
        results['active_location'] = active_water
        print(f"✓ Активна локація: {active_water}")
    
    # Глобальний таймер
    if coords.get('global_timer'):
        timer_img = pil_img.crop(coords['global_timer'])
        timer_img = apply_preprocessing(timer_img, preproc_settings)
        results['timer'] = pytesseract.image_to_string(timer_img, config=CONFIG_RUS).strip()
        print(f"✓ Таймер: {results['timer']}")
    
    # Обробка замовлень
    print("\n--- Аналіз замовлень ---")
    for i, order in enumerate(coords['orders']):
        order_result = {
            'card_num': i + 1,
            'fish_name': '',
            'timer': '',
            'quantity': '',
            'weight': '',
            'price': ''
        }
        
        card_img = pil_img.crop(order['full_box'])
        if not is_card_filled(card_img):
            print(f"\n[Замовлення {i+1}]: ПОРОЖНЯ")
            continue
        
        # Назва риби
        if i < len(coords.get('fish_names', [])):
            fish_img = pil_img.crop(coords['fish_names'][i])
            fish_img = apply_preprocessing(fish_img, preproc_settings)
            raw_text = pytesseract.image_to_string(fish_img, config=CONFIG_RUS).strip()
            cleaned_name = clean_fish_name(raw_text)

            # Видаляємо назву водойми, якщо вона є
            for water_body in WATER_BODIES:
                if water_body in cleaned_name:
                    cleaned_name = cleaned_name.replace(water_body, '').strip()

            order_result['fish_name'] = cleaned_name
        
        # Таймер картки
        if 'timer' in order:
            timer_img = pil_img.crop(order['timer'])
            timer_img = apply_preprocessing(timer_img, preproc_settings)
            raw_text = pytesseract.image_to_string(timer_img, config=CONFIG_RUS).strip()
            order_result['timer'] = clean_timer(raw_text)
        
        # Кількість
        if 'quantity' in order:
            qty_img = pil_img.crop(order['quantity'])
            qty_img = apply_preprocessing(qty_img, preproc_settings)
            raw_text = pytesseract.image_to_string(qty_img, config=CONFIG_RUS).strip()
            order_result['quantity'] = clean_quantity(raw_text)
        
        # Вага
        if 'weight' in order:
            weight_img = pil_img.crop(order['weight'])
            weight_img = apply_preprocessing(weight_img, preproc_settings)
            raw_text = pytesseract.image_to_string(weight_img, config=CONFIG_RUS).strip()
            order_result['weight'] = clean_weight(raw_text)
        
        # Ціна
        if 'price' in order:
            price_img = pil_img.crop(order['price'])
            price_settings = preproc_settings.copy()
            price_settings['scale_factor'] = 2.0
            price_img = apply_preprocessing(price_img, price_settings)
            
            config_price = '-l eng --psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.,'
            raw_text = pytesseract.image_to_string(price_img, config=config_price).strip()
            raw_text = raw_text.replace(' ', '').replace('O', '0').replace('o', '0')
            order_result['price'] = clean_price(raw_text)
        
        results['orders'].append(order_result)
        
        print(f"\n[Замовлення {i+1}]:")
        print(f"  Риба: {order_result['fish_name']}")
        print(f"  Таймер: {order_result['timer']}")
        print(f"  Кількість: {order_result['quantity']}")
        print(f"  Вага: {order_result['weight']}")
        print(f"  Ціна: {order_result['price']}")
    
    return results

def setup_specific_field_type(img_path, coords, field_type_choice):
    """Налаштування конкретного типу поля для всіх замовлень"""
    pil_img = normalize_image(img_path, TARGET_WIDTH, TARGET_HEIGHT)
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    if field_type_choice == '1':  # Заголовок
        roi = select_roi(cv_img, "Виділіть заголовок 'Кафе'")
        if roi:
            coords['header'] = roi
            print("✓ Заголовок оновлено")
    
    elif field_type_choice == '2':  # Водойми
        if 'water_bodies' not in coords:
            coords['water_bodies'] = {}
        print("Налаштування всіх 17 водойм...")
        water_bodies_names = [
            'оз. Комаринное', 'оз. Лосиное', 'р. Вьюнок', 'оз. Старый Острог', 'р. Белая',
            'оз. Куори', 'р. Волхов', 'р. Северный Донец', 'р. Сура', 'Ладожское оз.',
            'оз. Янтарное', 'Ладожский архипелаг', 'р. Ахтуба', 'оз. Медное',
            'р. Нижняя Тунгуска', 'р. Яма', 'Норвежское море'
        ]
        for water_name in water_bodies_names:
            roi = select_roi(cv_img, f"Виділіть: {water_name}")
            if roi:
                coords['water_bodies'][water_name] = roi
                print(f"✓ {water_name}")
    
    elif field_type_choice == '3':  # Глобальний таймер
        roi = select_roi(cv_img, "Виділіть глобальний таймер")
        if roi:
            coords['global_timer'] = roi
            print("✓ Глобальний таймер оновлено")
    
    elif field_type_choice == '4':  # Картки замовлень
        if 'orders' not in coords:
            coords['orders'] = []
        print("Налаштування 8 карток замовлень...")
        for i in range(8):
            roi = select_roi(cv_img, f"[{i+1}/8] Виділіть картку замовлення {i+1}")
            if roi:
                if i < len(coords['orders']):
                    coords['orders'][i]['full_box'] = roi
                else:
                    coords['orders'].append({'name': f'Замовлення {i+1}', 'full_box': roi})
                print(f"✓ Замовлення {i+1}")
    
    elif field_type_choice in ['5', '6', '7', '8', '9']:  # Поля на всіх картках
        field_name_map = {
            '5': ('fish_name', 'Назва риби'),
            '6': ('timer', 'Таймер'),
            '7': ('quantity', 'Кількість'),
            '8': ('weight', 'Вага'),
            '9': ('price', 'Ціна')
        }
        field_name, field_display = field_name_map[field_type_choice]
        
        print(f"\nНалаштування {field_display} на всіх картках...")
        
        if field_name == 'fish_name':
            if 'fish_names' not in coords:
                coords['fish_names'] = []
            
            for i, order in enumerate(coords.get('orders', [])):
                if 'full_box' not in order:
                    print(f"⚠ Замовлення {i+1} не має рамки, пропускаємо")
                    continue
                    
                x1, y1, x2, y2 = order['full_box']
                card_img = cv_img[y1:y2, x1:x2].copy()
                scale = 2
                card_big = cv2.resize(card_img, (card_img.shape[1]*scale, card_img.shape[0]*scale))
                
                roi = select_roi(card_big, f"[{i+1}/8] Назва риби на замовленні {i+1}")
                if roi:
                    fish_roi = (x1 + roi[0]//scale, y1 + roi[1]//scale, 
                               x1 + roi[2]//scale, y1 + roi[3]//scale)
                    
                    # Забезпечуємо що список достатньо великий
                    while len(coords['fish_names']) <= i:
                        coords['fish_names'].append(None)
                    coords['fish_names'][i] = fish_roi
                    print(f"✓ Замовлення {i+1}")
        else:
            # Для інших полів (timer, quantity, weight, price)
            for i, order in enumerate(coords.get('orders', [])):
                if 'full_box' not in order:
                    print(f"⚠ Замовлення {i+1} не має рамки, пропускаємо")
                    continue
                    
                x1, y1, x2, y2 = order['full_box']
                card_img = cv_img[y1:y2, x1:x2].copy()
                scale = 2
                card_big = cv2.resize(card_img, (card_img.shape[1]*scale, card_img.shape[0]*scale))
                
                roi = select_roi(card_big, f"[{i+1}/8] {field_display} на замовленні {i+1}")
                if roi:
                    order[field_name] = (x1 + roi[0]//scale, y1 + roi[1]//scale, 
                                        x1 + roi[2]//scale, y1 + roi[3]//scale)
                    print(f"✓ Замовлення {i+1}")
    
    return coords

def setup_specific_order(img_path, coords, order_num):
    """Налаштування конкретного замовлення"""
    pil_img = normalize_image(img_path, TARGET_WIDTH, TARGET_HEIGHT)
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    order_idx = order_num - 1
    
    # Забезпечуємо що є достатньо замовлень
    while len(coords.get('orders', [])) <= order_idx:
        coords.setdefault('orders', []).append({'name': f'Замовлення {len(coords.get("orders", []))+1}'})
    
    while True:
        choice = order_field_menu()
        
        if choice == '0':
            break
        elif choice == '1':  # Основна рамка
            roi = select_roi(cv_img, f"Виділіть рамку замовлення {order_num}")
            if roi:
                coords['orders'][order_idx]['full_box'] = roi
                print(f"✓ Рамка замовлення {order_num} оновлена")
        elif choice == '2':  # Назва риби
            if 'full_box' not in coords['orders'][order_idx]:
                print("⚠ Спочатку виділіть основну рамку картки!")
                continue
                
            x1, y1, x2, y2 = coords['orders'][order_idx]['full_box']
            card_img = cv_img[y1:y2, x1:x2].copy()
            scale = 2
            card_big = cv2.resize(card_img, (card_img.shape[1]*scale, card_img.shape[0]*scale))
            
            roi = select_roi(card_big, f"Назва риби на замовленні {order_num}")
            if roi:
                fish_roi = (x1 + roi[0]//scale, y1 + roi[1]//scale, 
                           x1 + roi[2]//scale, y1 + roi[3]//scale)
                
                # Забезпечуємо що список fish_names достатньо великий
                if 'fish_names' not in coords:
                    coords['fish_names'] = []
                while len(coords['fish_names']) <= order_idx:
                    coords['fish_names'].append(None)
                coords['fish_names'][order_idx] = fish_roi
                print(f"✓ Назва риби замовлення {order_num} оновлена")
        elif choice in ['3', '4', '5', '6']:  # Таймер, Кількість, Вага, Ціна
            if 'full_box' not in coords['orders'][order_idx]:
                print("⚠ Спочатку виділіть основну рамку картки!")
                continue
            
            field_map = {'3': ('timer', 'Таймер'), 
                        '4': ('quantity', 'Кількість'), 
                        '5': ('weight', 'Вага'), 
                        '6': ('price', 'Ціна')}
            field_name, field_display = field_map[choice]
            
            x1, y1, x2, y2 = coords['orders'][order_idx]['full_box']
            card_img = cv_img[y1:y2, x1:x2].copy()
            scale = 2
            card_big = cv2.resize(card_img, (card_img.shape[1]*scale, card_img.shape[0]*scale))
            
            roi = select_roi(card_big, f"{field_display} на замовленні {order_num}")
            if roi:
                coords['orders'][order_idx][field_name] = (
                    x1 + roi[0]//scale, y1 + roi[1]//scale, 
                    x1 + roi[2]//scale, y1 + roi[3]//scale
                )
                print(f"✓ {field_display} замовлення {order_num} оновлено")
    
    return coords

def interactive_full_setup(img_path):
    """Повне налаштування з нуля"""
    global roi_data
    
    print("\n" + "="*70)
    print("ПОВНЕ НАЛАШТУВАННЯ")
    print("="*70)
    
    pil_img = normalize_image(img_path, TARGET_WIDTH, TARGET_HEIGHT)
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    roi_data = {
        'header': None,
        'water_bodies': {},
        'global_timer': None,
        'orders': [],
        'fish_names': []
    }
    
    # 1. Заголовок
    print("\n--- Заголовок ---")
    roi = select_roi(cv_img, "Виділіть заголовок 'Кафе'")
    if roi:
        roi_data['header'] = roi
    
    # 2. Водойми
    print("\n--- Водойми (17 шт) ---")
    water_bodies_names = [
        'оз. Комаринное', 'оз. Лосиное', 'р. Вьюнок', 'оз. Старый Острог', 'р. Белая',
        'оз. Куори', 'р. Волхов', 'р. Северный Донец', 'р. Сура', 'Ладожское оз.',
        'оз. Янтарное', 'Ладожский архипелаг', 'р. Ахтуба', 'оз. Медное',
        'р. Нижняя Тунгуска', 'р. Яма', 'Норвежское море'
    ]
    for water_name in water_bodies_names:
        roi = select_roi(cv_img, f"Виділіть: {water_name}")
        if roi:
            roi_data['water_bodies'][water_name] = roi
    
    # 3. Глобальний таймер
    print("\n--- Глобальний таймер ---")
    roi = select_roi(cv_img, "Виділіть глобальний таймер")
    if roi:
        roi_data['global_timer'] = roi
    
    # 4. Картки замовлень
    print("\n--- 8 Карток замовлень ---")
    for i in range(8):
        roi = select_roi(cv_img, f"[{i+1}/8] Виділіть картку замовлення {i+1}")
        if roi:
            roi_data['orders'].append({'name': f'Замовлення {i+1}', 'full_box': roi})
    
    # 5. Назви риб
    print("\n--- Назви риб ---")
    for i, order in enumerate(roi_data['orders']):
        x1, y1, x2, y2 = order['full_box']
        card_img = cv_img[y1:y2, x1:x2].copy()
        scale = 2
        card_big = cv2.resize(card_img, (card_img.shape[1]*scale, card_img.shape[0]*scale))
        
        roi = select_roi(card_big, f"[{i+1}/8] Назва риби на картці {i+1}")
        if roi:
            fish_roi = (x1 + roi[0]//scale, y1 + roi[1]//scale, 
                       x1 + roi[2]//scale, y1 + roi[3]//scale)
            roi_data['fish_names'].append(fish_roi)
    
    # 6. Деталі карток
    print("\n--- Деталі карток ---")
    for i, order in enumerate(roi_data['orders']):
        print(f"\nКартка {i+1}:")
        x1, y1, x2, y2 = order['full_box']
        card_img = cv_img[y1:y2, x1:x2].copy()
        scale = 2
        card_big = cv2.resize(card_img, (card_img.shape[1]*scale, card_img.shape[0]*scale))
        
        # Таймер
        roi = select_roi(card_big, f"  Таймер на картці {i+1}")
        if roi:
            order['timer'] = (x1 + roi[0]//scale, y1 + roi[1]//scale, 
                            x1 + roi[2]//scale, y1 + roi[3]//scale)
        
        # Кількість
        roi = select_roi(card_big, f"  Кількість на картці {i+1}")
        if roi:
            order['quantity'] = (x1 + roi[0]//scale, y1 + roi[1]//scale, 
                               x1 + roi[2]//scale, y1 + roi[3]//scale)
        
        # Вага
        roi = select_roi(card_big, f"  Вага на картці {i+1}")
        if roi:
            order['weight'] = (x1 + roi[0]//scale, y1 + roi[1]//scale, 
                             x1 + roi[2]//scale, y1 + roi[3]//scale)
        
        # Ціна
        roi = select_roi(card_big, f"  Ціна на картці {i+1} (включіть кому!)")
        if roi:
            order['price'] = (x1 + roi[0]//scale, y1 + roi[1]//scale, 
                            x1 + roi[2]//scale, y1 + roi[3]//scale)
    
    return roi_data

# --- Функції для Debug режиму ---

def draw_debug_visualization(img_path, coords):
    """Візуалізує всі ROI області"""
    pil_img = normalize_image(img_path, TARGET_WIDTH, TARGET_HEIGHT)
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Заголовок (червоний)
    if coords.get('header'):
        x1, y1, x2, y2 = coords['header']
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(cv_img, 'HEADER', (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Водойми (помаранчевий)
    for name, roi in coords.get('water_bodies', {}).items():
        x1, y1, x2, y2 = roi
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 165, 255), 1)
    
    # Глобальний таймер (синій)
    if coords.get('global_timer'):
        x1, y1, x2, y2 = coords['global_timer']
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(cv_img, 'TIMER', (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Картки замовлень (зелені)
    for i, order in enumerate(coords.get('orders', [])):
        x1, y1, x2, y2 = order['full_box']
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(cv_img, f"Order {i+1}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Додаткові поля (блакитний)
        for field_name in ['timer', 'quantity', 'weight', 'price']:
            if field_name in order:
                fx1, fy1, fx2, fy2 = order[field_name]
                cv2.rectangle(cv_img, (fx1, fy1), (fx2, fy2), (255, 255, 0), 1)
    
    # Назви риб (жовті)
    for i, fish_roi in enumerate(coords.get('fish_names', [])):
        if fish_roi:
            x1, y1, x2, y2 = fish_roi
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(cv_img, f"Fish {i+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)
    
    return cv_img

def debug_mode_with_edit(img_path, coords, config_file):
    """Debug режим з можливістю редагування"""
    while True:
        # Показуємо візуалізацію
        debug_img = draw_debug_visualization(img_path, coords)
        cv2.imshow('Debug Visualization', debug_img)
        cv2.imwrite('debug_roi.png', debug_img)
        print("\n✓ Debug зображення збережено: debug_roi.png")
        print("Натисніть будь-яку клавішу для продовження...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Меню редагування
        choice = debug_menu()
        
        if choice == '0':
            break
        elif choice == '1':  # Редагувати тип полів
            field_choice = field_type_menu()
            if field_choice != '0':
                coords = setup_specific_field_type(img_path, coords, field_choice)
                save_coordinates(coords, config_file)
        elif choice == '2':  # Редагувати замовлення
            order_choice = order_menu()
            if order_choice != '0':
                order_num = int(order_choice)
                coords = setup_specific_order(img_path, coords, order_num)
                save_coordinates(coords, config_file)

# --- Головна функція ---

def main():
    parser = argparse.ArgumentParser(description="OCR з інтерактивним налаштуванням")
    parser.add_argument("image_path", help="Шлях до скріншота")
    parser.add_argument("--setup", action="store_true", help="Режим налаштування")
    parser.add_argument("--tune-preprocessing", action="store_true", help="Налаштування обробки")
    parser.add_argument("--debug", action="store_true", help="Візуалізація ROI з редагуванням")
    parser.add_argument("--config", default=CONFIG_FILE, help="Файл конфігурації")
    args = parser.parse_args()
    
    try:
        # Режим налаштування
        if args.setup:
            coords = load_coordinates(args.config)
            if coords is None:
                coords = {}
            
            while True:
                choice = setup_main_menu()
                
                if choice == '0':
                    break
                elif choice == '1':  # Повне налаштування
                    coords = interactive_full_setup(args.image_path)
                    if coords:
                        save_coordinates(coords, args.config)
                elif choice == '2':  # Редагувати тип поля
                    field_choice = field_type_menu()
                    if field_choice != '0':
                        coords = setup_specific_field_type(args.image_path, coords, field_choice)
                        save_coordinates(coords, args.config)
                elif choice == '3':  # Редагувати замовлення
                    order_choice = order_menu()
                    if order_choice != '0':
                        order_num = int(order_choice)
                        coords = setup_specific_order(args.image_path, coords, order_num)
                        save_coordinates(coords, args.config)
            return
        
        # Налаштування препроцесингу
        if args.tune_preprocessing:
            coords = load_coordinates(args.config)
            if not coords:
                print("⚠ Спочатку запустіть --setup")
                return
            
            print("\n" + "="*70)
            print("НАЛАШТУВАННЯ ОБРОБКИ - ВИБІР РЕЖИМУ")
            print("="*70)
            print("1 - Налаштувати для всіх замовлень")
            print("2 - Налаштувати для конкретного замовлення")
            mode_choice = input("Виберіть режим: ").strip()
            
            order_num = None
            if mode_choice == '2':
                order_num = int(input("Введіть номер замовлення (1-8): ").strip())
                if order_num < 1 or order_num > 8:
                    print("⚠ Неправильний номер замовлення")
                    return
            
            print("\nВиберіть тип поля для налаштування обробки:")
            print("1 - Заголовок (Кафе)")
            print("2 - Глобальний таймер")
            print("3 - Назва риби")
            print("4 - Таймер замовлення")
            print("5 - Кількість")
            print("6 - Вага")
            print("7 - Ціна")
            choice = input("Вибір: ").strip()
            
            field_map = {
                '1': 'header',
                '2': 'global_timer', 
                '3': 'fish_name', 
                '4': 'timer',
                '5': 'quantity', 
                '6': 'weight', 
                '7': 'price'
            }
            field_type = field_map.get(choice, 'price')
            
            settings = interactive_preprocessing_setup(args.image_path, coords, field_type, order_num)
            if settings:
                save_preprocessing_settings(settings)
            return
        
        # Debug режим з редагуванням
        if args.debug:
            coords = load_coordinates(args.config)
            if not coords:
                print("⚠ Спочатку запустіть --setup")
                return
            
            debug_mode_with_edit(args.image_path, coords, args.config)
            return
        
        # Завантаження конфігурації
        coords = load_coordinates(args.config)
        if not coords:
            print(f"\n⚠ Конфігурація не знайдена: {args.config}")
            print(f"Запустіть: python {os.path.basename(__file__)} {args.image_path} --setup")
            return
        
        preproc_settings = load_preprocessing_settings()
        
        # Звичайна обробка
        results = process_with_coordinates(args.image_path, coords, preproc_settings)
        
        # Збереження результатів
        output_file = 'ocr_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Результати збережено: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Помилка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
