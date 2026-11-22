//! nectar.exe: lend this PC's GPU, or borrow one from another PC on the network.
//!
//! - Borrow: lists the lenders nectar-svc finds, borrows or returns one, and shows the
//!   state (green: in use; amber: reconnecting; grey: no host) with the round trip and
//!   traffic. Programs then find the GPU on their own, through Nectar's drivers. Whichever it
//!   is for, gaming and rendering or AI and compute, tunes the link; Check link measures it.
//! - Lend: runs nectar-server --lend with the shared secret and shows its log.
//!
//! The tray icon shows the same state. Plain Win32 through windows-sys.

use crate::{lender, service};

use std::cell::{Cell, RefCell};
use std::ptr;

use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::{
    BITMAPINFO, BITMAPINFOHEADER, CreateBitmap, CreateDIBSection, CreateFontW, DIB_RGB_COLORS,
    DeleteObject, GetSysColorBrush, HDC, HFONT, InvalidateRect, SetBkMode, SetTextColor,
    TRANSPARENT,
    COLOR_WINDOW,
};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::Controls::{
    BST_CHECKED, EM_REPLACESEL, EM_SETSEL, ICC_LISTVIEW_CLASSES, INITCOMMONCONTROLSEX, InitCommonControlsEx, LVCF_TEXT, LVCF_WIDTH,
    LVCOLUMNW, LVIF_TEXT, LVITEMW, LVM_DELETEALLITEMS, LVM_GETNEXTITEM, LVM_INSERTCOLUMNW,
    LVM_INSERTITEMW, LVM_SETEXTENDEDLISTVIEWSTYLE, LVM_SETITEMSTATE, LVM_SETITEMTEXTW,
    LVIS_SELECTED, LVNI_SELECTED, LVS_EX_FULLROWSELECT, LVS_REPORT, LVS_SHOWSELALWAYS,
    LVS_SINGLESEL,
};
use windows_sys::Win32::UI::HiDpi::{
    DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, GetDpiForWindow, SetProcessDpiAwarenessContext,
};
use windows_sys::Win32::UI::Input::KeyboardAndMouse::EnableWindow;
use windows_sys::Win32::UI::Shell::{
    NIF_ICON, NIF_MESSAGE, NIF_TIP, NIM_ADD, NIM_DELETE, NIM_MODIFY, NOTIFYICONDATAW,
    Shell_NotifyIconW,
};
use windows_sys::Win32::UI::WindowsAndMessaging::*;

const ID_BORROW_MODE: u16 = 100;
const ID_LEND_MODE: u16 = 101;
const ID_LIST: u16 = 110;
const ID_BORROW: u16 = 111;
const ID_RETURN: u16 = 112;
const ID_ADDRESS: u16 = 113;
const ID_ADD: u16 = 114;
const ID_STATUS: u16 = 115;
const ID_STATS: u16 = 116;
const ID_HINT: u16 = 117;
const ID_SECRET_LABEL: u16 = 120;
const ID_SECRET: u16 = 121;
const ID_LEND: u16 = 122;
const ID_LEND_STATE: u16 = 123;
const ID_LOG: u16 = 124;
const ID_PROFILE_LABEL: u16 = 130;
const ID_GAMING: u16 = 131;
const ID_AI: u16 = 132;
const ID_CHECK: u16 = 133;
const ID_NOTE: u16 = 134;
const ID_REPORT: u16 = 135;

const TIMER: usize = 1;
const TRAY_MESSAGE: u32 = WM_APP + 1;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Light {
    Grey,
    Green,
    Amber,
    Blue,
}

struct App {
    window: HWND,
    lending_mode: bool,
    /// Controls of each mode, shown or hidden together
    borrow_controls: Vec<HWND>,
    lend_controls: Vec<HWND>,
    status: service::Status,
    /// Why the service could not be asked, shown instead of its status
    service_error: Option<String>,
    light: Light,
    icons: [HICON; 4],
    lender: Option<lender::Lender>,
    lend_state: String,
    /// The link check's text in the window, to change it only when it changes
    report: String,
    font: HFONT,
    big_font: HFONT,
}

thread_local! {
    static APP: RefCell<Option<App>> = const { RefCell::new(None) };
    /// The status light, apart from APP: setting a control's text makes Windows ask for
    /// its colour (WM_CTLCOLORSTATIC) while APP is in use
    static LIGHT: Cell<Light> = const { Cell::new(Light::Grey) };
}

/// Runs `f` on the app's state. Windows calls the window procedure again from inside many
/// calls f makes; such a nested call finds the state in use and does nothing.
fn with_app<R>(f: impl FnOnce(&mut App) -> R) -> Option<R> {
    APP.with(|app| app.try_borrow_mut().ok()?.as_mut().map(f))
}

fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(Some(0)).collect()
}

fn set_text(window: HWND, text: &str) {
    let text = wide(text);
    unsafe { SetWindowTextW(window, text.as_ptr()) };
}

fn get_text(window: HWND) -> String {
    let mut buffer = [0u16; 512];
    let length = unsafe { GetWindowTextW(window, buffer.as_mut_ptr(), buffer.len() as i32) };
    String::from_utf16_lossy(&buffer[..length.max(0) as usize])
}

fn control(parent: HWND, id: u16) -> HWND {
    unsafe { GetDlgItem(parent, id as i32) }
}

/// A filled circle in `color` (0xRRGGBB), for the tray and the window
fn dot_icon(color: u32) -> HICON {
    const SIZE: i32 = 32;
    let header = BITMAPINFOHEADER {
        biSize: size_of::<BITMAPINFOHEADER>() as u32,
        biWidth: SIZE,
        biHeight: -SIZE,
        biPlanes: 1,
        biBitCount: 32,
        biCompression: 0,
        biSizeImage: 0,
        biXPelsPerMeter: 0,
        biYPelsPerMeter: 0,
        biClrUsed: 0,
        biClrImportant: 0,
    };
    let info = BITMAPINFO {
        bmiHeader: header,
        bmiColors: [unsafe { std::mem::zeroed() }],
    };
    let mut bits: *mut std::ffi::c_void = ptr::null_mut();
    unsafe {
        let color_bitmap =
            CreateDIBSection(ptr::null_mut(), &info, DIB_RGB_COLORS, &mut bits, ptr::null_mut(), 0);
        let pixels = std::slice::from_raw_parts_mut(bits.cast::<u32>(), (SIZE * SIZE) as usize);
        let center = (SIZE as f32 - 1.0) / 2.0;
        for y in 0..SIZE {
            for x in 0..SIZE {
                let distance = ((x as f32 - center).powi(2) + (y as f32 - center).powi(2)).sqrt();
                // Edge softened over a pixel
                let coverage = (SIZE as f32 / 2.0 - 1.0 - distance).clamp(0.0, 1.0);
                let alpha = (coverage * 255.0) as u32;
                let scale = |channel: u32| channel * alpha / 255;
                let (r, g, b) = ((color >> 16) & 0xff, (color >> 8) & 0xff, color & 0xff);
                pixels[(y * SIZE + x) as usize] = (alpha << 24) | (scale(r) << 16) | (scale(g) << 8) | scale(b);
            }
        }
        let mask = CreateBitmap(SIZE, SIZE, 1, 1, ptr::null());
        let icon_info = ICONINFO {
            fIcon: 1,
            xHotspot: 0,
            yHotspot: 0,
            hbmMask: mask,
            hbmColor: color_bitmap,
        };
        let icon = CreateIconIndirect(&icon_info);
        DeleteObject(mask);
        DeleteObject(color_bitmap);
        icon
    }
}

fn tray(window: HWND, action: u32, icon: HICON, tip: &str) {
    let mut data: NOTIFYICONDATAW = unsafe { std::mem::zeroed() };
    data.cbSize = size_of::<NOTIFYICONDATAW>() as u32;
    data.hWnd = window;
    data.uID = 1;
    data.uFlags = NIF_ICON | NIF_MESSAGE | NIF_TIP;
    data.uCallbackMessage = TRAY_MESSAGE;
    data.hIcon = icon;
    for (slot, unit) in data.szTip.iter_mut().zip(tip.encode_utf16().take(127)) {
        *slot = unit;
    }
    unsafe { Shell_NotifyIconW(action, &data) };
}

pub fn main() {
    unsafe {
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        let controls = INITCOMMONCONTROLSEX {
            dwSize: size_of::<INITCOMMONCONTROLSEX>() as u32,
            dwICC: ICC_LISTVIEW_CLASSES,
        };
        InitCommonControlsEx(&controls);

        let instance = GetModuleHandleW(ptr::null());
        let class = wide("NectarWindow");
        let icon = dot_icon(0x8a8f98);
        let window_class = WNDCLASSW {
            style: CS_HREDRAW | CS_VREDRAW,
            lpfnWndProc: Some(window_proc),
            cbClsExtra: 0,
            cbWndExtra: 0,
            hInstance: instance,
            hIcon: icon,
            hCursor: LoadCursorW(ptr::null_mut(), IDC_ARROW),
            hbrBackground: GetSysColorBrush(COLOR_WINDOW),
            lpszMenuName: ptr::null(),
            lpszClassName: class.as_ptr(),
        };
        RegisterClassW(&window_class);

        let title = wide("Nectar");
        let window = CreateWindowExW(
            0,
            class.as_ptr(),
            title.as_ptr(),
            WS_OVERLAPPEDWINDOW & !WS_MAXIMIZEBOX & !WS_THICKFRAME,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            720,
            660,
            ptr::null_mut(),
            ptr::null_mut(),
            instance,
            ptr::null(),
        );
        ShowWindow(window, SW_SHOW);

        let mut message: MSG = std::mem::zeroed();
        while GetMessageW(&mut message, ptr::null_mut(), 0, 0) > 0 {
            if IsDialogMessageW(window, &message) == 0 {
                TranslateMessage(&message);
                DispatchMessageW(&message);
            }
        }
    }
}

unsafe fn create(window: HWND) {
    let instance = unsafe { GetModuleHandleW(ptr::null()) };
    let dpi = unsafe { GetDpiForWindow(window) }.max(96) as i32;
    let px = |value: i32| value * dpi / 96;
    let face = wide("Segoe UI");
    let font = unsafe {
        CreateFontW(-px(15), 0, 0, 0, 400, 0, 0, 0, 0, 0, 0, 5, 0, face.as_ptr())
    };
    let big_font = unsafe {
        CreateFontW(-px(20), 0, 0, 0, 600, 0, 0, 0, 0, 0, 0, 5, 0, face.as_ptr())
    };

    let add = |class: &str, text: &str, style: u32, x: i32, y: i32, w: i32, h: i32, id: u16| {
        let class = wide(class);
        let text = wide(text);
        let child = unsafe {
            CreateWindowExW(
                0,
                class.as_ptr(),
                text.as_ptr(),
                WS_CHILD | WS_VISIBLE | style,
                px(x),
                px(y),
                px(w),
                px(h),
                window,
                id as usize as HMENU,
                instance,
                ptr::null(),
            )
        };
        unsafe { SendMessageW(child, WM_SETFONT, font as WPARAM, 1) };
        child
    };

    let radio = (BS_AUTORADIOBUTTON | BS_PUSHLIKE) as u32 | WS_TABSTOP;
    add("BUTTON", "Borrow a GPU", radio | WS_GROUP, 16, 14, 150, 32, ID_BORROW_MODE);
    add("BUTTON", "Lend this GPU", radio, 172, 14, 150, 32, ID_LEND_MODE);

    let status = add("STATIC", "Asking the Nectar service...", 0, 16, 62, 670, 30, ID_STATUS);
    unsafe { SendMessageW(status, WM_SETFONT, big_font as WPARAM, 1) };
    let borrow_controls = vec![
        status,
        add("STATIC", "", 0, 16, 94, 670, 22, ID_STATS),
        add(
            "SysListView32",
            "",
            WS_BORDER | WS_TABSTOP | (LVS_REPORT | LVS_SINGLESEL | LVS_SHOWSELALWAYS),
            16,
            124,
            670,
            170,
            ID_LIST,
        ),
        add("BUTTON", "Borrow", WS_TABSTOP | BS_DEFPUSHBUTTON as u32, 16, 304, 110, 32, ID_BORROW),
        add("BUTTON", "Return", WS_TABSTOP, 134, 304, 110, 32, ID_RETURN),
        add("EDIT", "", WS_BORDER | WS_TABSTOP | ES_AUTOHSCROLL as u32, 386, 306, 190, 28, ID_ADDRESS),
        add("BUTTON", "Add address", WS_TABSTOP, 584, 304, 102, 32, ID_ADD),
        add("STATIC", "Use it for", 0, 16, 354, 70, 24, ID_PROFILE_LABEL),
        add("BUTTON", "Gaming && rendering", radio | WS_GROUP, 90, 348, 170, 32, ID_GAMING),
        add("BUTTON", "AI && compute", radio, 266, 348, 130, 32, ID_AI),
        add("BUTTON", "Check link", WS_TABSTOP | WS_GROUP, 584, 348, 102, 32, ID_CHECK),
        add("STATIC", "", 0, 16, 388, 670, 24, ID_NOTE),
        add(
            "EDIT",
            "",
            WS_BORDER | WS_VSCROLL | (ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL) as u32,
            16,
            416,
            670,
            136,
            ID_REPORT,
        ),
        add(
            "STATIC",
            "Programs find the borrowed GPU on their own. Lenders on this network appear here; \
             add an address if one does not.",
            0,
            16,
            560,
            670,
            44,
            ID_HINT,
        ),
    ];

    let list = control(window, ID_LIST);
    unsafe {
        SendMessageW(
            list,
            LVM_SETEXTENDEDLISTVIEWSTYLE,
            LVS_EX_FULLROWSELECT as WPARAM,
            LVS_EX_FULLROWSELECT as LPARAM,
        )
    };
    for (index, (title, width)) in [
        ("Lender", 130),
        ("GPU", 190),
        ("Memory", 70),
        ("Lends", 150),
        ("State", 80),
        ("Ping", 50),
    ]
    .into_iter()
    .enumerate()
    {
        let mut text = wide(title);
        let column = LVCOLUMNW {
            mask: LVCF_TEXT | LVCF_WIDTH,
            cx: px(width),
            pszText: text.as_mut_ptr(),
            ..unsafe { std::mem::zeroed() }
        };
        unsafe { SendMessageW(list, LVM_INSERTCOLUMNW, index, &column as *const _ as LPARAM) };
    }

    let saved = lender::saved_secret().unwrap_or_default();
    let lend_controls = vec![
        add("STATIC", "Shared secret", 0, 16, 66, 110, 24, ID_SECRET_LABEL),
        add("EDIT", &saved, WS_BORDER | WS_TABSTOP | (ES_PASSWORD | ES_AUTOHSCROLL) as u32, 130, 62, 260, 28, ID_SECRET),
        add("BUTTON", "Start lending", WS_TABSTOP, 400, 60, 150, 32, ID_LEND),
        add("STATIC", "Not lending", 0, 16, 104, 670, 26, ID_LEND_STATE),
        add(
            "EDIT",
            "",
            WS_BORDER | WS_VSCROLL | (ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL) as u32,
            16,
            136,
            670,
            468,
            ID_LOG,
        ),
    ];
    let state = control(window, ID_LEND_STATE);
    unsafe { SendMessageW(state, WM_SETFONT, big_font as WPARAM, 1) };

    let icons = [
        dot_icon(0x8a8f98),
        dot_icon(0x2e9e4f),
        dot_icon(0xe0a020),
        dot_icon(0x3478f6),
    ];
    tray(window, NIM_ADD, icons[0], "Nectar");

    APP.with(|app| {
        *app.borrow_mut() = Some(App {
            window,
            lending_mode: false,
            borrow_controls,
            lend_controls,
            status: service::Status::default(),
            service_error: None,
            light: Light::Grey,
            icons,
            lender: None,
            lend_state: String::new(),
            report: String::new(),
            font,
            big_font,
        })
    });
    unsafe {
        SendMessageW(control(window, ID_BORROW_MODE), BM_SETCHECK, BST_CHECKED as WPARAM, 0);
        SetTimer(window, TIMER, 1000, None);
    }
    show_mode(false);
    refresh();
}

fn show_mode(lending: bool) {
    with_app(|app| {
        app.lending_mode = lending;
        for &child in &app.borrow_controls {
            unsafe { ShowWindow(child, if lending { SW_HIDE } else { SW_SHOW }) };
        }
        for &child in &app.lend_controls {
            unsafe { ShowWindow(child, if lending { SW_SHOW } else { SW_HIDE }) };
        }
    });
}

/// Polls the service and the lender, and updates the window and the tray
fn refresh() {
    let polled = service::status();
    with_app(|app| {
        match polled {
            Ok(status) => {
                app.service_error = None;
                if status.found != app.status.found {
                    fill_list(app.window, &status.found);
                }
                app.status = status;
            }
            Err(error) => {
                app.service_error = Some(if error.kind() == std::io::ErrorKind::NotFound {
                    "The Nectar service is not running: install Nectar with Install-Borrower.ps1".to_string()
                } else {
                    format!("Cannot ask the Nectar service: {error}")
                });
                if !app.status.found.is_empty() {
                    fill_list(app.window, &[]);
                }
                app.status = service::Status::default();
            }
        }

        // The lender's new output, and what it says about the lease
        if let Some(lender) = &mut app.lender {
            let lines: Vec<String> = std::mem::take(
                &mut *lender.output.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
            );
            let log = control(app.window, ID_LOG);
            for line in &lines {
                append(log, line);
                // After the time lender.rs puts first
                let line = line.split_once(' ').map_or(line.as_str(), |(_, rest)| rest);
                if line.starts_with("lending ") {
                    app.lend_state = "Lending: ready for a borrower".into();
                } else if let Some(at) = line.find(" borrows this GPU") {
                    let who = line[..at].rsplit(' ').next().unwrap_or_default();
                    app.lend_state = format!("Lent to {who}");
                } else if line.contains("returned this GPU") {
                    app.lend_state = "Lending: ready for a borrower".into();
                }
            }
            if !lender.running() {
                app.lender = None;
                app.lend_state = "The server stopped; see the log".into();
                set_text(control(app.window, ID_LEND), "Start lending");
            }
        }
        set_text(
            control(app.window, ID_LEND_STATE),
            if app.lend_state.is_empty() { "Not lending" } else { &app.lend_state },
        );

        // The choice the service keeps, and what it means on this lender
        let gaming = app.status.profile == "gaming";
        let checked = |on: bool| if on { BST_CHECKED as WPARAM } else { 0 };
        unsafe {
            SendMessageW(control(app.window, ID_GAMING), BM_SETCHECK, checked(gaming), 0);
            SendMessageW(control(app.window, ID_AI), BM_SETCHECK, checked(app.status.profile == "ai"), 0);
            EnableWindow(control(app.window, ID_GAMING), app.service_error.is_none() as i32);
            EnableWindow(control(app.window, ID_AI), app.service_error.is_none() as i32);
            EnableWindow(
                control(app.window, ID_CHECK),
                (app.status.state == "borrowed" && app.status.check != "running") as i32,
            );
        }
        let note = if !app.status.note.is_empty() {
            app.status.note.clone()
        } else if app.status.state == "borrowed" && app.status.wifi_held {
            "Wi-Fi is in low-latency mode while the GPU is borrowed.".to_string()
        } else {
            match app.status.profile.as_str() {
                "gaming" => "Tuned for steady latency: little data waits in a queue.".to_string(),
                "ai" => "Tuned for throughput: results and models cross in bulk.".to_string(),
                _ => String::new(),
            }
        };
        set_text(control(app.window, ID_NOTE), &note);
        let report = report_text(&app.status);
        if report != app.report {
            set_text(control(app.window, ID_REPORT), &report);
            app.report = report;
        }

        let (text, stats, light) = match &app.service_error {
            Some(error) => (error.clone(), String::new(), Light::Grey),
            None => {
                let status = &app.status;
                let light = match status.state.as_str() {
                    "borrowed" => Light::Green,
                    "connecting" | "reconnecting" => Light::Amber,
                    _ => Light::Grey,
                };
                let stats = if status.state == "borrowed" {
                    let gpu = status.gpu_percent.map_or(String::new(), |percent| format!("GPU {percent}%   ·   "));
                    format!(
                        "{gpu}Round trip {:.1} ms   ·   {:.0} calls/s   ·   up {:.1} MB/s   ·   down {:.1} MB/s",
                        status.rtt_ms, status.calls_per_s, status.up_mb_s, status.down_mb_s
                    )
                } else {
                    String::new()
                };
                (status.message.clone(), stats, light)
            }
        };
        set_text(control(app.window, ID_STATUS), &text);
        set_text(control(app.window, ID_STATS), &stats);

        // Lending shows blue in the tray unless a borrowed GPU says more
        let light = if app.lender.is_some() && light == Light::Grey { Light::Blue } else { light };
        let tip = if app.lender.is_some() && light == Light::Blue {
            format!("Nectar: {}", app.lend_state)
        } else {
            format!("Nectar: {text}")
        };
        if light != app.light {
            app.light = light;
            LIGHT.set(light);
            unsafe { InvalidateRect(control(app.window, ID_STATUS), ptr::null(), 1) };
            let icon = app.icons[light as usize];
            unsafe {
                SendMessageW(app.window, WM_SETICON, ICON_SMALL as WPARAM, icon as LPARAM);
                SendMessageW(app.window, WM_SETICON, ICON_BIG as WPARAM, icon as LPARAM);
            }
        }
        tray(app.window, NIM_MODIFY, app.icons[app.light as usize], &tip);

        unsafe {
            let borrowing = !matches!(app.status.state.as_str(), "idle" | "");
            EnableWindow(control(app.window, ID_RETURN), borrowing as i32);
            EnableWindow(control(app.window, ID_BORROW), (app.service_error.is_none()) as i32);
        }
    });
}

/// The link check's findings, or how to get them
fn report_text(status: &service::Status) -> String {
    match status.check.as_str() {
        "running" => "Measuring the link for a few seconds...".to_string(),
        "" => "Check link measures the connection to the lender, and says what would help for what \
               you chose above."
            .to_string(),
        _ => status
            .check_lines
            .iter()
            .chain(&status.advice)
            .cloned()
            .collect::<Vec<_>>()
            .join("\r\n"),
    }
}

fn fill_list(window: HWND, found: &[service::Lender]) {
    let list = control(window, ID_LIST);
    let selected = selected_index(list);
    unsafe { SendMessageW(list, LVM_DELETEALLITEMS, 0, 0) };
    for (row, lender) in found.iter().enumerate() {
        let state = if lender.state == "lent" {
            format!("lent to {}", lender.borrower)
        } else {
            "ready".to_string()
        };
        let cells = [
            lender.name.clone(),
            lender.gpu.clone(),
            format!("{:.1} GB", lender.vram_mib.parse::<f64>().unwrap_or(0.0) / 1024.0),
            lender.apis.clone(),
            state,
            format!("{} ms", lender.rtt_ms),
        ];
        for (column, cell) in cells.iter().enumerate() {
            let mut text = wide(cell);
            let item = LVITEMW {
                mask: LVIF_TEXT,
                iItem: row as i32,
                iSubItem: column as i32,
                pszText: text.as_mut_ptr(),
                ..unsafe { std::mem::zeroed() }
            };
            let message = if column == 0 { LVM_INSERTITEMW } else { LVM_SETITEMTEXTW };
            unsafe { SendMessageW(list, message, row, &item as *const _ as LPARAM) };
        }
    }
    if let Some(index) = selected.filter(|&index| index < found.len()).or((!found.is_empty()).then_some(0)) {
        let item = LVITEMW {
            stateMask: LVIS_SELECTED,
            state: LVIS_SELECTED,
            ..unsafe { std::mem::zeroed() }
        };
        unsafe { SendMessageW(list, LVM_SETITEMSTATE, index, &item as *const _ as LPARAM) };
    }
}

fn selected_index(list: HWND) -> Option<usize> {
    let index = unsafe { SendMessageW(list, LVM_GETNEXTITEM, usize::MAX, LVNI_SELECTED as LPARAM) };
    (index >= 0).then_some(index as usize)
}

fn append(log: HWND, line: &str) {
    let text = wide(&format!("{line}\r\n"));
    unsafe {
        let length = GetWindowTextLengthW(log);
        SendMessageW(log, EM_SETSEL, length as WPARAM, length as LPARAM);
        SendMessageW(log, EM_REPLACESEL, 0, text.as_ptr() as LPARAM);
    }
}

fn message_box(window: HWND, text: &str) {
    let text = wide(text);
    let title = wide("Nectar");
    unsafe { MessageBoxW(window, text.as_ptr(), title.as_ptr(), MB_OK | MB_ICONINFORMATION) };
}

fn command(window: HWND, id: u16) {
    match id {
        ID_BORROW_MODE => show_mode(false),
        ID_LEND_MODE => show_mode(true),
        ID_BORROW => {
            let chosen = with_app(|app| {
                selected_index(control(window, ID_LIST))
                    .and_then(|index| app.status.found.get(index))
                    .map(|lender| lender.name.clone())
            })
            .flatten();
            match chosen {
                Some(name) => {
                    if let Err(error) = service::ask(&format!("borrow {name}")) {
                        message_box(window, &format!("Cannot borrow from {name}: {error}"));
                    }
                }
                None => message_box(window, "Pick a lender in the list first."),
            }
            refresh();
        }
        ID_RETURN => {
            let _ = service::ask("return");
            refresh();
        }
        ID_ADD => {
            let address = get_text(control(window, ID_ADDRESS)).trim().to_string();
            if !address.is_empty() {
                if let Err(error) = service::ask(&format!("add {address}")) {
                    message_box(window, &format!("Cannot add {address}: {error}"));
                }
                set_text(control(window, ID_ADDRESS), "");
            }
        }
        ID_GAMING | ID_AI => {
            let profile = if id == ID_GAMING { "gaming" } else { "ai" };
            if let Err(error) = service::ask(&format!("profile {profile}")) {
                message_box(window, &format!("Cannot change it: {error}"));
            }
            refresh();
        }
        ID_CHECK => {
            if let Err(error) = service::ask("check") {
                message_box(window, &format!("Cannot check the link: {error}"));
            }
            refresh();
        }
        ID_LEND => toggle_lending(window),
        _ => {}
    }
}

fn toggle_lending(window: HWND) {
    let lending = with_app(|app| app.lender.is_some()).unwrap_or(false);
    if lending {
        with_app(|app| {
            if let Some(mut lender) = app.lender.take() {
                lender.stop();
            }
            app.lend_state = "Not lending".into();
            append(control(window, ID_LOG), "Stopped lending.");
        });
        set_text(control(window, ID_LEND), "Start lending");
        return;
    }

    let secret = get_text(control(window, ID_SECRET)).trim().to_string();
    if secret.len() < 8 {
        message_box(
            window,
            "Choose a shared secret of at least 8 characters, and give the same one to Install-Borrower.ps1 on the laptop.",
        );
        return;
    }
    match lender::Lender::start(&secret) {
        Ok(started) => {
            lender::save_secret(&secret);
            with_app(|app| {
                app.lender = Some(started);
                app.lend_state = "Starting...".into();
                let log = control(window, ID_LOG);
                append(log, "Starting nectar-server --lend...");
                append(
                    log,
                    "If borrowers cannot find this PC, Windows Firewall needs to let them in on TCP 43212 and UDP 43211, on Private networks: Start-Lender.ps1 prints the commands.",
                );
            });
            set_text(control(window, ID_LEND), "Stop lending");
        }
        Err(error) => message_box(window, &format!("Cannot start lending: {error}")),
    }
    refresh();
}

unsafe extern "system" fn window_proc(window: HWND, message: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    match message {
        WM_CREATE => {
            unsafe { create(window) };
            0
        }
        WM_TIMER => {
            refresh();
            0
        }
        WM_COMMAND => {
            command(window, (wparam & 0xffff) as u16);
            0
        }
        WM_CTLCOLORSTATIC => {
            let target = lparam as HWND;
            let dc = wparam as HDC;
            let color = (target == control(window, ID_STATUS)).then(|| match LIGHT.get() {
                Light::Green => 0x4f9e2e,
                Light::Amber => 0x1080c0,
                Light::Blue => 0xf67834,
                Light::Grey => 0x707070,
            });
            unsafe {
                if let Some(color) = color {
                    // COLORREF is 0x00BBGGRR
                    SetTextColor(dc, color);
                }
                SetBkMode(dc, TRANSPARENT as i32);
                GetSysColorBrush(COLOR_WINDOW) as LRESULT
            }
        }
        TRAY_MESSAGE => {
            if lparam as u32 == WM_LBUTTONUP {
                unsafe {
                    ShowWindow(window, SW_RESTORE);
                    SetForegroundWindow(window);
                }
            }
            0
        }
        WM_DESTROY => {
            with_app(|app| {
                let lending = app.lender.take();
                drop(lending);
                tray(app.window, NIM_DELETE, ptr::null_mut(), "");
                unsafe {
                    DeleteObject(app.font);
                    DeleteObject(app.big_font);
                }
            });
            unsafe { PostQuitMessage(0) };
            0
        }
        _ => unsafe { DefWindowProcW(window, message, wparam, lparam) },
    }
}

