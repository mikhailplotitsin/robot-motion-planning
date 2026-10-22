#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <array>
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <windows.h>
#include <clocale>

using namespace std;

const double PI = 3.14159265358979323846;

struct DHParam {
    double a;
    double alpha;
    double d;
    double theta_offset;
};

const DHParam DH[6] = {
    {0.0,     PI / 2,   0.037, PI / 2},
    {0.063,   0.0,      0.0,   PI / 2},
    {0.0,     PI / 2,   0.0,   PI / 2},
    {0.0,    -PI / 2,   0.235, PI / 9},
    {0.0,     0.0,      0.0,   7 * PI / 12},
    {0.0,     0.0,      0.028, PI / 2}
};

const double Q_MIN[6] = { 0.0, PI / 3, PI / 18, 0.0, PI / 2, 0.0 };
const double Q_MAX[6] = { PI, 17 * PI / 18, PI / 6, PI, 2 * PI / 3, PI };

using Matrix4 = array<array<double, 4>, 4>;
using Vector6 = array<double, 6>;

double deg2rad(double d) { return d * PI / 180.0; }
double rad2deg(double r) { return r * 180.0 / PI; }

Matrix4 multiply(const Matrix4& A, const Matrix4& B) {
    Matrix4 C{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

double read_double(const string& prompt) {
    double v;
    cout << prompt;
    cin >> v;
    return v;
}

void input_data(Vector6& q_init, array<double, 3>& A, array<double, 3>& B) {
    cout << string(60, '=') << "\n";
    cout << "  ПРОГРАММА ПЛАНИРОВАНИЯ ДВИЖЕНИЯ 6-ОСЕВОГО РОБОТА\n";
    cout << string(60, '=') << "\n";

    cout << "\n[БЛОК 1] НАЧАЛЬНОЕ ПОЛОЖЕНИЕ (6 углов, градусы)\n";
    cout << "  Ограничения: q1[0..180], q2[60..170], q3[10..30], q4[0..180], q5[90..120], q6[0..180]\n";
    for (int i = 0; i < 6; ++i) {
        ostringstream oss;
        oss << "  q" << (i + 1) << "_init = ";
        double val = read_double(oss.str());
        q_init[i] = deg2rad(val);
    }

    cout << "\n[БЛОК 1] КООРДИНАТЫ ТОЧЕК A и B (метры)\n";
    cout << "  Точка A:\n";
    A[0] = read_double("    xA = ");
    A[1] = read_double("    yA = ");
    A[2] = read_double("    zA = ");

    cout << "  Точка B:\n";
    B[0] = read_double("    xB = ");
    B[1] = read_double("    yB = ");
    B[2] = read_double("    zB = ");
}

Matrix4 dh_matrix(double a, double alpha, double d, double theta) {
    double ct = cos(theta), st = sin(theta);
    double ca = cos(alpha), sa = sin(alpha);
    return Matrix4{ {
        {ct, -st * ca,  st * sa, a * ct},
        {st,  ct * ca, -ct * sa, a * st},
        {0.0, sa,     ca,    d   },
        {0.0, 0.0,    0.0,   1.0 }
    } };
}

Matrix4 forward_kinematics(const Vector6& q) {
    Matrix4 T = { {{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1}} };
    for (int i = 0; i < 6; ++i) {
        Matrix4 Ti = dh_matrix(DH[i].a, DH[i].alpha, DH[i].d, q[i] + DH[i].theta_offset);
        T = multiply(T, Ti);
    }
    return T;
}

void get_position(const Vector6& q, double& x, double& y, double& z) {
    Matrix4 T = forward_kinematics(q);
    x = T[0][3];
    y = T[1][3];
    z = T[2][3];
}

// ──────────────────────────────────────────────────────────────────
//  РЕШЕНИЕ ЛИНЕЙНОЙ СИСТЕМЫ A·x = b (метод Гаусса)
// ──────────────────────────────────────────────────────────────────
bool solve_linear_system(vector<vector<double>>& A, vector<double>& b, vector<double>& x) {
    int n = (int)A.size();
    x.resize(n);
    for (int i = 0; i < n; ++i) {
        int max_row = i;
        for (int k = i + 1; k < n; ++k)
            if (abs(A[k][i]) > abs(A[max_row][i])) max_row = k;
        swap(A[i], A[max_row]);
        swap(b[i], b[max_row]);
        if (abs(A[i][i]) < 1e-12) return false;
        for (int k = i + 1; k < n; ++k) {
            double factor = A[k][i] / A[i][i];
            for (int j = i; j < n; ++j) A[k][j] -= factor * A[i][j];
            b[k] -= factor * b[i];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0;
        for (int j = i + 1; j < n; ++j) sum += A[i][j] * x[j];
        x[i] = (b[i] - sum) / A[i][i];
    }
    return true;
}

// ──────────────────────────────────────────────────────────────────
//  ОЗК: Левенберг–Марквардт с мультистартом
// ──────────────────────────────────────────────────────────────────
bool inverse_kinematics(array<double, 3> target, Vector6& q_out) {
    double global_best_err = 1e9;
    const double TOL = 0.001;   // 1 мм
    const int    TRIALS = 20;
    const int    MAX_ITER = 500;
    const double DQ_STEP = 1e-6;

    for (int trial = 0; trial < TRIALS; ++trial) {
        Vector6 q;
        if (trial == 0) {
            for (int i = 0; i < 6; ++i) q[i] = (Q_MIN[i] + Q_MAX[i]) / 2.0;
        }
        else {
            for (int i = 0; i < 6; ++i) {
                double rnd = (double)rand() / RAND_MAX;
                q[i] = Q_MIN[i] + rnd * (Q_MAX[i] - Q_MIN[i]);
            }
        }

        double lambda = 0.1;
        for (int iter = 0; iter < MAX_ITER; ++iter) {
            double cx, cy, cz;
            get_position(q, cx, cy, cz);
            double ex = target[0] - cx;
            double ey = target[1] - cy;
            double ez = target[2] - cz;
            double err = sqrt(ex * ex + ey * ey + ez * ez);

            if (err < TOL) { q_out = q; return true; }
            if (err < global_best_err) global_best_err = err;

            double J[3][6];
            for (int j = 0; j < 6; ++j) {
                Vector6 q_temp = q; q_temp[j] += DQ_STEP;
                double px, py, pz; get_position(q_temp, px, py, pz);
                J[0][j] = (px - cx) / DQ_STEP;
                J[1][j] = (py - cy) / DQ_STEP;
                J[2][j] = (pz - cz) / DQ_STEP;
            }

            vector<vector<double>> A(6, vector<double>(6, 0.0));
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j < 6; ++j)
                    A[i][j] = J[0][i] * J[0][j] + J[1][i] * J[1][j] + J[2][i] * J[2][j];
                A[i][i] += lambda;
            }
            vector<double> b(6, 0.0);
            for (int i = 0; i < 6; ++i)
                b[i] = J[0][i] * ex + J[1][i] * ey + J[2][i] * ez;

            vector<double> delta_q;
            if (!solve_linear_system(A, b, delta_q)) break;

            Vector6 q_new;
            for (int i = 0; i < 6; ++i) {
                q_new[i] = q[i] + delta_q[i];
                if (q_new[i] < Q_MIN[i]) q_new[i] = Q_MIN[i];
                if (q_new[i] > Q_MAX[i]) q_new[i] = Q_MAX[i];
            }

            double cx2, cy2, cz2; get_position(q_new, cx2, cy2, cz2);
            double err_new = sqrt(pow(target[0] - cx2, 2) + pow(target[1] - cy2, 2) + pow(target[2] - cz2, 2));

            if (err_new < err) {
                q = q_new; lambda *= 0.5; if (lambda < 1e-5) lambda = 1e-5;
            }
            else {
                lambda *= 2.0; if (lambda > 1e5) lambda = 1e5;
            }
        }
    }
    cout << "  [DEBUG] Минимальное расстояние до цели: " << global_best_err * 1000.0 << " мм\n";
    return false;
}

bool validate_angles(const Vector6& q) {
    for (int i = 0; i < 6; ++i) {
        if (q[i] < Q_MIN[i] - 1e-5 || q[i] > Q_MAX[i] + 1e-5) {
            cout << "  ! q" << i + 1 << " = " << rad2deg(q[i]) << " град вне qlim\n";
            return false;
        }
    }
    return true;
}

void print_pose(const string& label, const Vector6& q) {
    Matrix4 T = forward_kinematics(q);
    cout << "\n[ПЗК для " << label << "]\n";
    cout << fixed << setprecision(2);
    cout << "  q = [";
    for (int i = 0; i < 6; ++i) cout << rad2deg(q[i]) << (i < 5 ? ", " : "");
    cout << "] град\n";
    cout << setprecision(3);
    cout << "  Положение: X = " << T[0][3] * 1000 << " мм, Y = " << T[1][3] * 1000
        << " мм, Z = " << T[2][3] * 1000 << " мм\n";
}

// ==================== РАБОТА С ПОСЛЕДОВАТЕЛЬНЫМ ПОРТОМ ====================

HANDLE open_serial(const string& port, int baud = 115200) {
    string full = "\\\\.\\" + port;
    HANDLE h = CreateFileA(full.c_str(), GENERIC_READ | GENERIC_WRITE,
        0, NULL, OPEN_EXISTING, 0, NULL);
    if (h == INVALID_HANDLE_VALUE) return INVALID_HANDLE_VALUE;

    DCB dcb = { 0 };
    dcb.DCBlength = sizeof(dcb);
    GetCommState(h, &dcb);
    dcb.BaudRate = baud;
    dcb.ByteSize = 8;
    dcb.StopBits = ONESTOPBIT;
    dcb.Parity = NOPARITY;
    dcb.fBinary = TRUE;
    dcb.fParity = FALSE;
    dcb.fOutxCtsFlow = FALSE;
    dcb.fOutxDsrFlow = FALSE;
    dcb.fDtrControl = DTR_CONTROL_ENABLE;
    dcb.fRtsControl = RTS_CONTROL_ENABLE;
    dcb.fOutX = FALSE;
    dcb.fInX = FALSE;
    SetCommState(h, &dcb);

    COMMTIMEOUTS to = { 0 };
    to.ReadIntervalTimeout = 20;
    to.ReadTotalTimeoutConstant = 100;
    to.ReadTotalTimeoutMultiplier = 10;
    to.WriteTotalTimeoutConstant = 100;
    to.WriteTotalTimeoutMultiplier = 10;
    SetCommTimeouts(h, &to);
    return h;
}

bool send_line(HANDLE h, const string& line) {
    DWORD written = 0;
    return WriteFile(h, line.c_str(), (DWORD)line.size(), &written, NULL)
        && written == (DWORD)line.size();
}

bool send_angles(HANDLE h, const Vector6& q) {
    ostringstream oss;
    oss << fixed << setprecision(2);
    for (int i = 0; i < 6; ++i) {
        oss << rad2deg(q[i]);
        if (i < 5) oss << ",";
    }
    oss << "\n";
    return send_line(h, oss.str());
}

void drain_input(HANDLE h) {
    char buf[256]; DWORD n = 0;
    while (ReadFile(h, buf, sizeof(buf) - 1, &n, NULL) && n > 0) { /* discard */ }
}

bool send_angles_wait_ack(HANDLE h, const Vector6& q, DWORD timeout_ms = 2000) {
    drain_input(h);
    if (!send_angles(h, q)) return false;

    string r;
    DWORD start = GetTickCount();
    while (GetTickCount() - start < timeout_ms) {
        char buf[256]; DWORD n = 0;
        if (ReadFile(h, buf, sizeof(buf) - 1, &n, NULL) && n > 0) {
            buf[n] = 0; r += buf;
            if (r.find("OK") != string::npos || r.find("INIT_DONE") != string::npos) return true;
            if (r.find("ERR") != string::npos) return false;
        }
        Sleep(10);
    }
    return false;
}

// ⚠️ НОВАЯ ФУНКЦИЯ: Отправка команды магнита
bool send_magnet_command(HANDLE h, const string& cmd, DWORD timeout_ms = 2000) {
    drain_input(h);
    if (!send_line(h, cmd + "\n")) return false;

    string r;
    DWORD start = GetTickCount();
    while (GetTickCount() - start < timeout_ms) {
        char buf[256]; DWORD n = 0;
        if (ReadFile(h, buf, sizeof(buf) - 1, &n, NULL) && n > 0) {
            buf[n] = 0; r += buf;
            if (r.find("MAG_OK") != string::npos) return true;
        }
        Sleep(10);
    }
    return false;
}

Vector6 lerp(const Vector6& a, const Vector6& b, double t) {
    Vector6 r;
    for (int i = 0; i < 6; ++i) r[i] = a[i] + t * (b[i] - a[i]);
    return r;
}

void send_trajectory(HANDLE h, const Vector6& from, const Vector6& to, int steps, int step_ms) {
    for (int i = 1; i <= steps; ++i) {
        Vector6 q = lerp(from, to, (double)i / steps);
        if (send_angles_wait_ack(h, q, 3000)) {
            cout << "  [" << setw(2) << i << "/" << steps << "] q = [";
            for (int j = 0; j < 6; ++j) {
                cout << fixed << setprecision(1) << rad2deg(q[j]);
                if (j < 5) cout << ",";
            }
            cout << "]\n";
        }
        else {
            cout << "  ! нет ACK от Arduino на шаге " << i << "\n";
        }
        Sleep(step_ms);
    }
}

// ==================== MAIN ====================
int main() {
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);
    setlocale(LC_ALL, "Russian");

    srand((unsigned)time(nullptr));   // для мультистарта ОЗК

    Vector6 q_init;
    array<double, 3> A, B;
    input_data(q_init, A, B);

    cout << "\n[БЛОК 2] ВАЛИДАЦИЯ\n";
    if (!validate_angles(q_init)) {
        cout << "  ! Ошибка в начальных углах. Остановка программы.\n";
        return 1;
    }
    cout << "  OK! Начальные углы в пределах qlim\n";

    cout << "\n[БЛОК 3] КИНЕМАТИКА (ОЗК)\n";
    Vector6 qA, qB;

    cout << "  Решение ОЗК для точки A (Левенберг–Марквардт, 20 стартов)...\n";
    if (!inverse_kinematics(A, qA)) {
        cout << "  Ошибка! Точка A недостижима (ОЗК не нашла решения)\n";
        return 1;
    }
    cout << "  OK! ОЗК для A найдена\n";

    cout << "  Решение ОЗК для точки B (Левенберг–Марквардт, 20 стартов)...\n";
    if (!inverse_kinematics(B, qB)) {
        cout << "  Ошибка! Точка B недостижима (ОЗК не нашла решения)\n";
        return 1;
    }
    cout << "  OK! ОЗК для B найдена\n";

    cout << "\n[БЛОК 4] ПЛАНИРОВАНИЕ ТРАЕКТОРИИ\n";
    cout << "  q_init -> qA: 10 waypoints\n";
    cout << "  qA -> qB:     10 waypoints\n";

    cout << "\n" << string(60, '=') << "\n";
    cout << "  РЕЗУЛЬТАТЫ РАСЧЁТА\n";
    cout << string(60, '=') << "\n";

    print_pose("НАЧАЛЬНЫХ УСЛОВИЙ", q_init);
    print_pose("ТОЧКИ A (после ОЗК)", qA);
    print_pose("ТОЧКИ B (после ОЗК)", qB);

    cout << "\n" << string(60, '=') << "\n";
    cout << "  ИТОГОВЫЙ ОТВЕТ\n";
    cout << string(60, '=') << "\n\n";

    Matrix4 T = forward_kinematics(q_init);
    cout << fixed << setprecision(3);
    cout << "  ПЗК для начальных условий = ("
        << T[0][3] << ", " << T[1][3] << ", " << T[2][3] << ") м\n";

    cout << "\n  q_init = [";
    for (int i = 0; i < 6; ++i) cout << setprecision(2) << rad2deg(q_init[i]) << (i < 5 ? ", " : "");
    cout << "] град\n";

    // ==================== БЛОК 5: ОТПРАВКА НА ARDUINO ====================
    cout << "\n" << string(60, '=') << "\n";
    cout << "  БЛОК 5: ОТПРАВКА УГЛОВ НА ARDUINO\n";
    cout << string(60, '=') << "\n";

    cout << "\n  Отправлять данные на Arduino? (1 - да, 0 - нет): ";
    int do_send = 0;
    cin >> do_send;
    if (do_send != 1) {
        cout << "  Отправка пропущена. Программа завершена.\n";
        return 0;
    }

    string port;
    cout << "  COM-порт (например COM3): ";
    cin >> port;

    HANDLE hSerial = open_serial(port, 115200);
    if (hSerial == INVALID_HANDLE_VALUE) {
        cout << "  ! Не удалось открыть " << port << ". Проверьте номер порта и что он не занят.\n";
        return 1;
    }
    cout << "  OK! Порт " << port << " открыт\n";

    cout << "  Ожидание инициализации Arduino (2 с)...\n";
    Sleep(2000);
    drain_input(hSerial);

    cout << "\n  Отправка начального положения...\n";
    if (send_angles_wait_ack(hSerial, q_init, 3000)) {
        cout << "  OK! q_init принят\n";
    }
    else {
        cout << "  ! нет ACK на q_init\n";
    }
    Sleep(500);

    // --- 1. ДВИЖЕНИЕ К ТОЧКЕ А ---
    cout << "\n  Движение q_init -> qA (10 waypoints):\n";
    send_trajectory(hSerial, q_init, qA, 10, 250);
    Sleep(500);

    // --- 2. ВКЛЮЧЕНИЕ МАГНИТА И ПАУЗА ---
    cout << "\n  Прибытие в точку А. Включение электромагнита...\n";
    if (send_magnet_command(hSerial, "MAG_ON")) {
        cout << "  OK! Электромагнит включен\n";
    }
    else {
        cout << "  ! Ошибка включения магнита\n";
    }

    cout << "  Пауза 7 секунд для захвата объекта...\n";
    Sleep(7000);

    // --- 3. ДВИЖЕНИЕ К ТОЧКЕ Б ---
    cout << "\n  Движение qA -> qB (10 waypoints):\n";
    send_trajectory(hSerial, qA, qB, 10, 250);
    Sleep(500);

    // --- 4. ВЫКЛЮЧЕНИЕ МАГНИТА ---
    cout << "\n  Прибытие в точку Б. Выключение электромагнита...\n";
    if (send_magnet_command(hSerial, "MAG_OFF")) {
        cout << "  OK! Электромагнит выключен\n";
    }
    else {
        cout << "  ! Ошибка выключения магнита\n";
    }

    CloseHandle(hSerial);
    cout << "\n  Готово. Порт закрыт.\n";

    return 0;
}