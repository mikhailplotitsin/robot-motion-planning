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

struct DHParam { double a, alpha, d, theta_offset; };

const DHParam DH[6] = {
    {0.0,     PI / 2,   0.037, PI / 2},
    {0.063,   0.0,      0.0,   PI / 2},
    {0.0,     PI / 2,   0.0,   PI / 2},
    {0.0,    -PI / 2,   0.235, PI / 9},
    {0.0,     0.0,      0.0,   7 * PI / 12},
    {0.0,     0.0,      0.028, PI / 2}
};

const double Q_MIN[6] = { 0.0, PI / 3, PI / 18, 0.0, PI / 2, 0.0 };
const double Q_MAX[6] = { PI,  17 * PI / 18, PI / 6, PI, 2 * PI / 3, PI };

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
    double v; cout << prompt; cin >> v; return v;
}

void input_data(Vector6& q_init, array<double, 3>& A, array<double, 3>& B,
    double& q1_des_A, double& q1_des_B)
{
    cout << string(60, '=') << "\n";
    cout << "  ПЛАНИРОВАНИЕ ДВИЖЕНИЯ 6-ОСЕВОГО РОБОТА\n";
    cout << "  (ОЗК с посевом по желаемому q1)\n";
    cout << string(60, '=') << "\n";

    cout << "\n[БЛОК 1] НАЧАЛЬНОЕ ПОЛОЖЕНИЕ (6 углов, градусы)\n";
    cout << "  Ограничения: q1[0..180], q2[60..170], q3[10..30], q4[0..180], q5[90..120], q6[0..180]\n";
    for (int i = 0; i < 6; ++i) {
        ostringstream oss; oss << "  q" << (i + 1) << "_init = ";
        q_init[i] = deg2rad(read_double(oss.str()));
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

    // ─── Желаемый q1: используется как seed для ОЗК ─────────────────
    cout << "\n[БЛОК 1] ЖЕЛАЕМЫЙ ПЕРВЫЙ УГОЛ q1 (град)\n";
    cout << "  ОЗК будет искать решение, ближайшее к этому q1:\n";
    q1_des_A = read_double("    q1_desired для A = ");
    q1_des_B = read_double("    q1_desired для B = ");
}

Matrix4 dh_matrix(double a, double alpha, double d, double theta) {
    double ct = cos(theta), st = sin(theta);
    double ca = cos(alpha), sa = sin(alpha);
    return Matrix4{ {
        {ct, -st * ca,  st * sa, a * ct},
        {st,  ct * ca, -ct * sa, a * st},
        {0.0, sa,       ca,      d   },
        {0.0, 0.0,      0.0,     1.0 }
    } };
}

Matrix4 forward_kinematics(const Vector6& q) {
    Matrix4 T = { {
        {1,0,0,0},
        {0,1,0,0},
        {0,0,1,0},
        {0,0,0,1}
    } };
    for (int i = 0; i < 6; ++i)
        T = multiply(T, dh_matrix(DH[i].a, DH[i].alpha, DH[i].d,
            q[i] + DH[i].theta_offset));
    return T;
}

void get_position(const Vector6& q, double& x, double& y, double& z) {
    Matrix4 T = forward_kinematics(q);
    x = T[0][3]; y = T[1][3]; z = T[2][3];
}

// ─── Гаусс ─────────────────────────────────────────────────────────
bool solve_linear_system(vector<vector<double>>& A, vector<double>& b, vector<double>& x) {
    int n = (int)A.size(); x.resize(n);
    for (int i = 0; i < n; ++i) {
        int mr = i;
        for (int k = i + 1; k < n; ++k)
            if (abs(A[k][i]) > abs(A[mr][i])) mr = k;
        swap(A[i], A[mr]); swap(b[i], b[mr]);
        if (abs(A[i][i]) < 1e-12) return false;
        for (int k = i + 1; k < n; ++k) {
            double f = A[k][i] / A[i][i];
            for (int j = i; j < n; ++j) A[k][j] -= f * A[i][j];
            b[k] -= f * b[i];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = 0;
        for (int j = i + 1; j < n; ++j) s += A[i][j] * x[j];
        x[i] = (b[i] - s) / A[i][i];
    }
    return true;
}

// ─── ОЗК: Левенберг–Марквардт, seed передаётся снаружи ──────────────
bool inverse_kinematics(array<double, 3> target, const Vector6& q_seed, Vector6& q_out) {
    double global_best_err = 1e9;
    const double TOL = 0.001;
    const int    TRIALS = 20;
    const int    MAX_ITER = 500;
    const double DQ_STEP = 1e-6;

    for (int trial = 0; trial < TRIALS; ++trial) {
        Vector6 q;
        if (trial == 0) {
            q = q_seed;                 // старт из пользовательского seed
        }
        else {
            for (int i = 0; i < 6; ++i) {
                double rnd = (double)rand() / RAND_MAX;
                q[i] = Q_MIN[i] + rnd * (Q_MAX[i] - Q_MIN[i]);
            }
        }

        double lambda = 0.1;
        for (int iter = 0; iter < MAX_ITER; ++iter) {
            double cx, cy, cz; get_position(q, cx, cy, cz);
            double ex = target[0] - cx;
            double ey = target[1] - cy;
            double ez = target[2] - cz;
            double err = sqrt(ex * ex + ey * ey + ez * ez);

            if (err < TOL) { q_out = q; return true; }
            if (err < global_best_err) global_best_err = err;

            double J[3][6];
            for (int j = 0; j < 6; ++j) {
                Vector6 qt = q; qt[j] += DQ_STEP;
                double px, py, pz; get_position(qt, px, py, pz);
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

            vector<double> dq;
            if (!solve_linear_system(A, b, dq)) break;

            Vector6 qn;
            for (int i = 0; i < 6; ++i) {
                qn[i] = q[i] + dq[i];
                if (qn[i] < Q_MIN[i]) qn[i] = Q_MIN[i];
                if (qn[i] > Q_MAX[i]) qn[i] = Q_MAX[i];
            }
            double nx, ny, nz; get_position(qn, nx, ny, nz);
            double err_new = sqrt(pow(target[0] - nx, 2) + pow(target[1] - ny, 2) + pow(target[2] - nz, 2));
            if (err_new < err) { q = qn; lambda = max(1e-5, lambda * 0.5); }
            else { lambda = min(1e5, lambda * 2.0); }
        }
    }
    cout << "  [DEBUG] Минимальное расстояние до цели: "
        << fixed << setprecision(2) << global_best_err * 1000.0 << " мм\n";
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

// ─── Диагностика: q1 фактический vs желаемый + отклонение от seed ───
void print_q1_report(const string& label, const Vector6& q,
    const Vector6& seed, double q1_desired_deg)
{
    double q1_fact_deg = rad2deg(q[0]);
    double q1_dev = abs(q1_fact_deg - q1_desired_deg);
    cout << fixed << setprecision(2);
    cout << "  " << label << ": q1_fact = " << q1_fact_deg
        << "°, q1_desired = " << q1_desired_deg
        << "°, |Δq1| = " << q1_dev << "°\n";

    double max_dev = 0; int max_idx = 0;
    for (int i = 0; i < 6; ++i) {
        double d = abs(rad2deg(q[i] - seed[i]));
        if (d > max_dev) { max_dev = d; max_idx = i; }
    }
    cout << "  " << label << ": макс. отклонение от seed = "
        << max_dev << "° (q" << (max_idx + 1) << ")\n";

    if (q1_dev > 5.0) {
        cout << "  ! q1 ушёл от желаемого на " << q1_dev
            << "° — точка может быть недостижима при этом q1\n";
        cout << "  ! (или ОЗК соскочила на другую ветвь; попробуйте другой q1)\n";
    }
    if (max_dev > 10.0)
        cout << "  ! Решение далеко от seed — возможна другая ветвь\n";
}

void verify_position(const char* name, const Vector6& q, const array<double, 3>& target) {
    double x, y, z; get_position(q, x, y, z);
    double err_mm = sqrt(pow(x - target[0], 2) + pow(y - target[1], 2) + pow(z - target[2], 2)) * 1000.0;
    cout << fixed << setprecision(3);
    cout << "  [" << name << "] ПЗК(q) = (" << x << ", " << y << ", " << z << ") м\n";
    cout << "  [" << name << "] |ПЗК(q) − target| = " << err_mm << " мм"
        << (err_mm < 1.0 ? "  OK" : "  !!! ПРЕВЫШЕН ДОПУСК") << "\n";
}

// ==================== ПОСЛЕДОВАТЕЛЬНЫЙ ПОРТ ====================
HANDLE open_serial(const string& port, int baud = 115200) {
    string full = "\\\\.\\" + port;
    HANDLE h = CreateFileA(full.c_str(), GENERIC_READ | GENERIC_WRITE,
        0, NULL, OPEN_EXISTING, 0, NULL);
    if (h == INVALID_HANDLE_VALUE) return INVALID_HANDLE_VALUE;
    DCB dcb = { 0 }; dcb.DCBlength = sizeof(dcb);
    GetCommState(h, &dcb);
    dcb.BaudRate = baud; dcb.ByteSize = 8; dcb.StopBits = ONESTOPBIT;
    dcb.Parity = NOPARITY; dcb.fBinary = TRUE; dcb.fParity = FALSE;
    dcb.fOutxCtsFlow = FALSE; dcb.fOutxDsrFlow = FALSE;
    dcb.fDtrControl = DTR_CONTROL_ENABLE; dcb.fRtsControl = RTS_CONTROL_ENABLE;
    dcb.fOutX = FALSE; dcb.fInX = FALSE;
    SetCommState(h, &dcb);
    COMMTIMEOUTS to = { 0 };
    to.ReadIntervalTimeout = 20; to.ReadTotalTimeoutConstant = 100;
    to.ReadTotalTimeoutMultiplier = 10; to.WriteTotalTimeoutConstant = 100;
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
    ostringstream oss; oss << fixed << setprecision(2);
    for (int i = 0; i < 6; ++i) { oss << rad2deg(q[i]); if (i < 5) oss << ","; }
    oss << "\n"; return send_line(h, oss.str());
}
void drain_input(HANDLE h) {
    char buf[256]; DWORD n = 0;
    while (ReadFile(h, buf, sizeof(buf) - 1, &n, NULL) && n > 0) {}
}
bool send_angles_wait_ack(HANDLE h, const Vector6& q, DWORD timeout_ms = 2000) {
    drain_input(h);
    if (!send_angles(h, q)) return false;
    string r; DWORD start = GetTickCount();
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
bool send_magnet_command(HANDLE h, const string& cmd, DWORD timeout_ms = 2000) {
    drain_input(h);
    if (!send_line(h, cmd + "\n")) return false;
    string r; DWORD start = GetTickCount();
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
    Vector6 r; for (int i = 0; i < 6; ++i) r[i] = a[i] + t * (b[i] - a[i]); return r;
}
void send_trajectory(HANDLE h, const Vector6& from, const Vector6& to,
    int steps, int step_ms) {
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
    SetConsoleCP(1251); SetConsoleOutputCP(1251);
    setlocale(LC_ALL, "Russian");
    srand((unsigned)time(nullptr));

    Vector6 q_init;
    array<double, 3> A, B;
    double q1_des_A = 0, q1_des_B = 0;
    input_data(q_init, A, B, q1_des_A, q1_des_B);

    cout << "\n[БЛОК 2] ВАЛИДАЦИЯ\n";
    if (!validate_angles(q_init)) {
        cout << "  ! Ошибка в начальных углах. Остановка программы.\n";
        return 1;
    }
    // Проверка, что желаемые q1 в пределах qlim
    double q1A_rad = deg2rad(q1_des_A);
    double q1B_rad = deg2rad(q1_des_B);
    if (q1A_rad < Q_MIN[0] - 1e-5 || q1A_rad > Q_MAX[0] + 1e-5) {
        cout << "  ! q1_desired для A = " << q1_des_A
            << "° вне [" << rad2deg(Q_MIN[0]) << ".." << rad2deg(Q_MAX[0]) << "]\n";
        return 1;
    }
    if (q1B_rad < Q_MIN[0] - 1e-5 || q1B_rad > Q_MAX[0] + 1e-5) {
        cout << "  ! q1_desired для B = " << q1_des_B
            << "° вне [" << rad2deg(Q_MIN[0]) << ".." << rad2deg(Q_MAX[0]) << "]\n";
        return 1;
    }
    cout << "  OK! Углы в пределах qlim\n";

    // ─── БЛОК 3: ОЗК с пользовательским q1 ─────────────────────────
    cout << "\n[БЛОК 3] КИНЕМАТИКА (ОЗК, seed с желаемым q1)\n";
    Vector6 qA, qB;

    // Seed A = q_init, но q1 заменяем на желаемый
    Vector6 seedA = q_init;
    seedA[0] = q1A_rad;
    cout << "  Поиск qA (seed = q_init с q1 = " << q1_des_A << "°)...\n";
    if (!inverse_kinematics(A, seedA, qA)) {
        cout << "  Ошибка! Точка A недостижима\n";
        return 1;
    }
    cout << "  OK! qA найдено\n";
    print_q1_report("A", qA, seedA, q1_des_A);

    // Seed B = qA, но q1 заменяем на желаемый
    Vector6 seedB = qA;
    seedB[0] = q1B_rad;
    cout << "  Поиск qB (seed = qA с q1 = " << q1_des_B << "°)...\n";
    if (!inverse_kinematics(B, seedB, qB)) {
        cout << "  Ошибка! Точка B недостижима\n";
        return 1;
    }
    cout << "  OK! qB найдено\n";
    print_q1_report("B", qB, seedB, q1_des_B);

    // ─── БЛОК 4: проверка через ПЗК ────────────────────────────────
    cout << "\n[БЛОК 4] ПРОВЕРКА ЧЕРЕЗ ПЗК\n";
    verify_position("A", qA, A);
    verify_position("B", qB, B);

    // ─── Результаты ────────────────────────────────────────────────
    cout << "\n" << string(60, '=') << "\n  РЕЗУЛЬТАТЫ РАСЧЁТА\n"
        << string(60, '=') << "\n";
    print_pose("НАЧАЛЬНЫХ УСЛОВИЙ", q_init);
    print_pose("ТОЧКИ A (после ОЗК)", qA);
    print_pose("ТОЧКИ B (после ОЗК)", qB);

    // ─── БЛОК 5: Arduino ───────────────────────────────────────────
    cout << "\n" << string(60, '=') << "\n  БЛОК 5: ОТПРАВКА НА ARDUINO\n"
        << string(60, '=') << "\n";
    cout << "\n  Отправлять данные на Arduino? (1 - да, 0 - нет): ";
    int do_send = 0; cin >> do_send;
    if (do_send != 1) { cout << "  Отправка пропущена.\n"; return 0; }

    string port;
    cout << "  COM-порт (например COM3): "; cin >> port;

    HANDLE hSerial = open_serial(port, 115200);
    if (hSerial == INVALID_HANDLE_VALUE) {
        cout << "  ! Не удалось открыть " << port << "\n"; return 1;
    }
    cout << "  OK! Порт " << port << " открыт\n";
    cout << "  Ожидание инициализации Arduino (2 с)...\n";
    Sleep(2000); drain_input(hSerial);

    cout << "\n  Отправка начального положения...\n";
    if (send_angles_wait_ack(hSerial, q_init, 3000))
        cout << "  OK! q_init принят\n";
    else
        cout << "  ! нет ACK на q_init\n";
    Sleep(500);

    cout << "\n  Движение q_init -> qA (10 waypoints):\n";
    send_trajectory(hSerial, q_init, qA, 10, 250);
    Sleep(500);

    cout << "\n  Прибытие в точку A. Включение электромагнита...\n";
    if (send_magnet_command(hSerial, "MAG_ON"))
        cout << "  OK! Электромагнит включен\n";
    else
        cout << "  ! Ошибка включения магнита\n";

    cout << "  Пауза 7 секунд для захвата объекта...\n";
    Sleep(7000);

    cout << "\n  Движение qA -> qB (10 waypoints):\n";
    send_trajectory(hSerial, qA, qB, 10, 250);
    Sleep(500);

    cout << "\n  Прибытие в точку B. Выключение электромагнита...\n";
    if (send_magnet_command(hSerial, "MAG_OFF"))
        cout << "  OK! Электромагнит выключен\n";
    else
        cout << "  ! Ошибка выключения магнита\n";

    CloseHandle(hSerial);
    cout << "\n  Готово. Порт закрыт.\n";
    return 0;
}